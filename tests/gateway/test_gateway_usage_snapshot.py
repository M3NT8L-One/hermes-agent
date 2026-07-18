from types import SimpleNamespace

from gateway.run import _agent_usage_snapshot


def test_agent_usage_snapshot_preserves_canonical_token_buckets():
    agent = SimpleNamespace(
        session_input_tokens=30_000,
        session_prompt_tokens=90_000,
        session_cache_read_tokens=50_000,
        session_cache_write_tokens=10_000,
        session_completion_tokens=10_000,
        session_total_tokens=100_000,
        context_compressor=SimpleNamespace(
            last_prompt_tokens=18_600,
            context_length=372_000,
        ),
    )

    assert _agent_usage_snapshot(agent) == {
        "input_tokens": 30_000,
        "prompt_tokens": 90_000,
        "cache_read_tokens": 50_000,
        "cache_write_tokens": 10_000,
        "output_tokens": 10_000,
        "total_tokens": 100_000,
        "last_prompt_tokens": 18_600,
        "context_length": 372_000,
    }
