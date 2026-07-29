"""Per-session API agent admission regressions.

These tests exercise the decorator directly so ordering is deterministic and no
real model/provider is involved.
"""

import asyncio
import queue
import uuid
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import gateway.platforms.api_server as api_server_module
from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _admit_api_agent_request,
    _cancel_and_wait_task_terminal,
)


class _Request:
    def __init__(self, *, session_id="", body=None, path="/v1/chat/completions"):
        self.headers = {"Authorization": "Bearer test-key"}
        if session_id:
            self.headers["X-Hermes-Session-Id"] = session_id
        self.match_info = {}
        self._body = dict(body or {})
        self.path = path
        self.method = "POST"
        self.remote = "127.0.0.1"

    async def json(self):
        return dict(self._body)


def _adapter(key=""):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": key}))
    # These tests do not exercise response persistence. Close the constructor's
    # SQLite handle immediately so repeated focused runs are descriptor-neutral.
    adapter._response_store.close()
    return adapter


def test_cancel_helper_waits_for_child_terminal_cleanup():
    async def exercise():
        started = asyncio.Event()
        cleaned_up = asyncio.Event()

        async def child():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                cleaned_up.set()

        task = asyncio.create_task(child())
        await started.wait()
        await _cancel_and_wait_task_terminal(task)
        assert task.done()
        assert cleaned_up.is_set()

    asyncio.run(exercise())


def test_chat_sse_cancellation_waits_for_agent_task_terminal():
    async def exercise():
        adapter = _adapter()
        write_started = asyncio.Event()
        cleanup_started = asyncio.Event()
        allow_cleanup = asyncio.Event()

        class BlockingStreamResponse:
            def __init__(self, **_kwargs):
                pass

            async def prepare(self, _request):
                return None

            async def write(self, _payload):
                write_started.set()
                await asyncio.Event().wait()

        async def child():
            try:
                await asyncio.Event().wait()
            finally:
                cleanup_started.set()
                await allow_cleanup.wait()

        agent_task = asyncio.create_task(child())
        with patch.object(api_server_module.web, "StreamResponse", BlockingStreamResponse):
            writer = asyncio.create_task(
                adapter._write_sse_chat_completion(
                    _Request(), "chatcmpl_test", "test-model", 1,
                    queue.Queue(), agent_task,
                )
            )
            await asyncio.wait_for(write_started.wait(), timeout=1.0)
            writer.cancel()
            await asyncio.wait_for(cleanup_started.wait(), timeout=1.0)
            await asyncio.sleep(0.02)
            assert not writer.done()
            allow_cleanup.set()
            try:
                await asyncio.wait_for(writer, timeout=1.0)
            except asyncio.CancelledError:
                pass
        assert agent_task.done()

    asyncio.run(exercise())


def test_responses_sse_cancellation_waits_for_agent_task_terminal():
    async def exercise():
        adapter = _adapter()
        write_started = asyncio.Event()
        cleanup_started = asyncio.Event()
        allow_cleanup = asyncio.Event()

        class BlockingStreamResponse:
            def __init__(self, **_kwargs):
                pass

            async def prepare(self, _request):
                return None

            async def write(self, _payload):
                write_started.set()
                await asyncio.Event().wait()

        async def child():
            try:
                await asyncio.Event().wait()
            finally:
                cleanup_started.set()
                await allow_cleanup.wait()

        agent_task = asyncio.create_task(child())
        with patch.object(api_server_module.web, "StreamResponse", BlockingStreamResponse):
            writer = asyncio.create_task(
                adapter._write_sse_responses(
                    _Request(), "resp_test", "test-model", 1,
                    queue.Queue(), agent_task, [], [], "hello",
                    None, None, False, "shared",
                )
            )
            await asyncio.wait_for(write_started.wait(), timeout=1.0)
            writer.cancel()
            await asyncio.wait_for(cleanup_started.wait(), timeout=1.0)
            await asyncio.sleep(0.02)
            assert not writer.done()
            allow_cleanup.set()
            try:
                await asyncio.wait_for(writer, timeout=1.0)
            except asyncio.CancelledError:
                pass
        assert agent_task.done()

    asyncio.run(exercise())


def test_cancelled_stream_holds_admission_until_child_cleanup_finishes():
    async def exercise():
        adapter = _adapter()
        child_started = asyncio.Event()
        cleanup_started = asyncio.Event()
        allow_cleanup = asyncio.Event()
        wake_entered = asyncio.Event()

        async def child():
            child_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleanup_started.set()
                await allow_cleanup.wait()

        @_admit_api_agent_request
        async def handler(self, request):
            if request._body["name"] == "stream":
                agent_task = asyncio.create_task(child())
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    await _cancel_and_wait_task_terminal(agent_task)
                    raise
            wake_entered.set()
            return "wake"

        stream = asyncio.create_task(
            handler(adapter, _Request(session_id="shared", body={"name": "stream"}))
        )
        await asyncio.wait_for(child_started.wait(), timeout=1.0)
        stream.cancel()
        await asyncio.wait_for(cleanup_started.wait(), timeout=1.0)

        wake = asyncio.create_task(
            handler(adapter, _Request(session_id="shared", body={"name": "wake"}))
        )
        try:
            await asyncio.wait_for(wake_entered.wait(), timeout=0.03)
        except asyncio.TimeoutError:
            pass
        else:
            raise AssertionError("wake entered before cancelled stream child was terminal")

        allow_cleanup.set()
        try:
            await stream
        except asyncio.CancelledError:
            pass
        assert await asyncio.wait_for(wake, timeout=1.0) == "wake"
        assert adapter._session_agent_admissions == {}

    asyncio.run(exercise())


def test_same_header_session_waits_before_handler_mutation():
    async def exercise():
        adapter = _adapter()
        first_entered = asyncio.Event()
        release_first = asyncio.Event()
        second_entered = asyncio.Event()
        mutations = []

        @_admit_api_agent_request
        async def handler(self, request):
            name = request._body["name"]
            mutations.append(name)
            if name == "first":
                first_entered.set()
                await release_first.wait()
            else:
                second_entered.set()
            return name

        first = asyncio.create_task(
            handler(adapter, _Request(session_id="shared", body={"name": "first"}))
        )
        await first_entered.wait()
        second = asyncio.create_task(
            handler(adapter, _Request(session_id="shared", body={"name": "wake"}))
        )

        try:
            await asyncio.wait_for(second_entered.wait(), timeout=0.03)
        except asyncio.TimeoutError:
            pass
        else:
            raise AssertionError("same-session wake entered before foreground turn ended")

        assert mutations == ["first"]
        release_first.set()
        assert await asyncio.gather(first, second) == ["first", "wake"]
        assert mutations == ["first", "wake"]
        assert adapter._session_agent_admissions == {}

    asyncio.run(exercise())


def test_runs_body_session_id_participates_in_same_admission_lane():
    async def exercise():
        adapter = _adapter()
        first_entered = asyncio.Event()
        release_first = asyncio.Event()
        second_entered = asyncio.Event()

        @_admit_api_agent_request
        async def handler(self, request):
            name = request._body["name"]
            if name == "first":
                first_entered.set()
                await release_first.wait()
            else:
                second_entered.set()
            return name

        first = asyncio.create_task(
            handler(
                adapter,
                _Request(
                    path="/v1/runs",
                    body={"session_id": "shared-run", "name": "first"},
                ),
            )
        )
        await first_entered.wait()
        second = asyncio.create_task(
            handler(
                adapter,
                _Request(
                    path="/v1/runs",
                    body={"session_id": "shared-run", "name": "wake"},
                ),
            )
        )

        try:
            await asyncio.wait_for(second_entered.wait(), timeout=0.03)
        except asyncio.TimeoutError:
            pass
        else:
            raise AssertionError("same-session /v1/runs requests overlapped")

        release_first.set()
        assert await asyncio.gather(first, second) == ["first", "wake"]

    asyncio.run(exercise())


def test_different_sessions_are_not_globally_serialized():
    async def exercise():
        adapter = _adapter()
        first_entered = asyncio.Event()
        release_first = asyncio.Event()
        other_entered = asyncio.Event()

        @_admit_api_agent_request
        async def handler(self, request):
            if request._body["name"] == "first":
                first_entered.set()
                await release_first.wait()
            else:
                other_entered.set()
            return request._body["name"]

        first = asyncio.create_task(
            handler(adapter, _Request(session_id="one", body={"name": "first"}))
        )
        await first_entered.wait()
        other = asyncio.create_task(
            handler(adapter, _Request(session_id="two", body={"name": "other"}))
        )
        await asyncio.wait_for(other_entered.wait(), timeout=0.1)
        release_first.set()
        assert await asyncio.gather(first, other) == ["first", "other"]

    asyncio.run(exercise())


def test_background_run_keeps_session_admission_until_task_finishes():
    async def exercise():
        adapter = _adapter()
        background_started = asyncio.Event()
        release_background = asyncio.Event()
        wake_entered = asyncio.Event()

        @_admit_api_agent_request
        async def handler(self, request):
            if request._body["name"] == "run":
                async def background():
                    background_started.set()
                    await release_background.wait()

                self._activate_admitted_request()
                task = asyncio.create_task(background())
                self._detach_admitted_request_to_task(task)
                return "started"
            wake_entered.set()
            return "wake"

        assert await handler(
            adapter,
            _Request(session_id="shared", body={"name": "run"}, path="/v1/runs"),
        ) == "started"
        await background_started.wait()
        wake = asyncio.create_task(
            handler(adapter, _Request(session_id="shared", body={"name": "wake"}))
        )

        try:
            await asyncio.wait_for(wake_entered.wait(), timeout=0.03)
        except asyncio.TimeoutError:
            pass
        else:
            raise AssertionError("wake entered while detached /v1/runs task was active")

        release_background.set()
        assert await wake == "wake"
        assert adapter._session_agent_admissions == {}

    asyncio.run(exercise())


async def _async_value(value):
    return value


def test_incident_repro_foreground_then_idempotent_wake_retry_runs_once():
    """Reproduce foreground overlap plus an ambiguous completion-wake retry."""
    async def exercise():
        adapter = _adapter("test-key")
        setattr(adapter, "_ensure_session_db_async", lambda: _async_value(None))
        foreground_started = asyncio.Event()
        release_foreground = asyncio.Event()
        calls = []

        async def run_agent(*, user_message: str, session_id: str, **_kwargs):
            calls.append(user_message)
            if user_message == "foreground":
                foreground_started.set()
                await release_foreground.wait()
            return (
                {"final_response": f"done:{user_message}", "session_id": session_id},
                {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            )

        setattr(adapter, "_run_agent", run_agent)
        app = web.Application()
        app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
        headers = {
            "Authorization": "Bearer test-key",
            "X-Hermes-Session-Id": "incident-session",
        }
        wake_key = f"hermes-wake-v1:async_delegation:{uuid.uuid4().hex}"

        async with TestClient(TestServer(app)) as client:
            foreground = asyncio.create_task(
                client.post(
                    "/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": "foreground"}],
                    },
                )
            )
            await asyncio.wait_for(foreground_started.wait(), timeout=1.0)

            wake = asyncio.create_task(
                client.post(
                    "/v1/chat/completions",
                    headers={**headers, "Idempotency-Key": wake_key},
                    json={
                        "model": "hermes-agent",
                        "messages": [
                            {"role": "user", "content": "completion callback"}
                        ],
                    },
                )
            )

            async def wake_is_queued():
                while adapter._session_agent_admissions["incident-session"]["refs"] < 2:
                    await asyncio.sleep(0)

            await asyncio.wait_for(wake_is_queued(), timeout=1.0)
            assert calls == ["foreground"]

            release_foreground.set()
            first_response, wake_response = await asyncio.gather(foreground, wake)
            assert first_response.status == 200
            assert wake_response.status == 200
            assert calls == ["foreground", "completion callback"]

            retry_response = await client.post(
                "/v1/chat/completions",
                headers={**headers, "Idempotency-Key": wake_key},
                json={
                    "model": "hermes-agent",
                    "messages": [
                        {"role": "user", "content": "completion callback"}
                    ],
                },
            )
            assert retry_response.status == 200
            assert calls == ["foreground", "completion callback"]

    asyncio.run(exercise())
