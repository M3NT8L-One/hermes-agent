"""Tests for state.db ownership and long-lived runtime identity diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import runtime_ownership


def _lsof_file(
    *,
    pid: int,
    command: str,
    fd: str,
    inode: int,
    links: int,
    path: str,
    deleted: bool = False,
) -> str:
    suffix = " (deleted)" if deleted else ""
    return f"p{pid}\nc{command}\nf{fd}\ni{inode}\nk{links}\nn{path}{suffix}\n"


def test_parse_lsof_field_output_preserves_inode_link_and_deleted_state():
    parsed = runtime_ownership.parse_lsof_field_output(
        _lsof_file(
            pid=42,
            command="python",
            fd="11u",
            inode=987,
            links=0,
            path="/tmp/state.db-wal",
            deleted=True,
        )
    )

    assert len(parsed) == 1
    item = parsed[0]
    assert item.pid == 42
    assert item.fd == "11u"
    assert item.inode == 987
    assert item.link_count == 0
    assert item.path == "/tmp/state.db-wal"
    assert item.deleted is True


def test_inspection_detects_split_and_unlinked_wal_generation(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    db_path.write_bytes(b"db")
    Path(f"{db_path}-wal").write_bytes(b"wal")
    Path(f"{db_path}-shm").write_bytes(b"shm")

    linked = _lsof_file(
        pid=100,
        command="python",
        fd="9u",
        inode=1000,
        links=1,
        path=str(db_path),
    ) + _lsof_file(
        pid=100,
        command="python",
        fd="10u",
        inode=2000,
        links=1,
        path=f"{db_path}-wal",
    )
    unlinked = _lsof_file(
        pid=200,
        command="python",
        fd="12u",
        inode=2001,
        links=0,
        path=f"{db_path}-wal",
        deleted=True,
    )
    outputs = iter((linked, unlinked))

    def fake_runner(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=next(outputs), stderr="")

    monkeypatch.setattr(
        runtime_ownership.shutil, "which", lambda _name: "/usr/sbin/lsof"
    )
    monkeypatch.setattr(
        runtime_ownership,
        "_full_process_command",
        lambda pid, fallback: f"{fallback} --pid {pid}",
    )

    report = runtime_ownership.inspect_state_db_ownership(db_path, runner=fake_runner)

    assert report.available is True
    assert report.open_inodes("wal") == {2000, 2001}
    assert report.split_kinds == {"wal": {2000, 2001}}
    assert [(item.pid, item.inode) for item in report.unlinked_files] == [(200, 2001)]


def test_lsof_no_match_is_available_but_actual_error_is_not(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    db_path.write_bytes(b"db")
    monkeypatch.setattr(
        runtime_ownership.shutil, "which", lambda _name: "/usr/sbin/lsof"
    )
    commands: list[list[str]] = []

    def no_match_runner(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=1, stdout="", stderr="")

    no_match = runtime_ownership.inspect_state_db_ownership(
        db_path,
        runner=no_match_runner,
    )
    assert no_match.available is True
    assert no_match.files == ()
    assert commands == [
        ["/usr/sbin/lsof", "-nP", "-Fpcfikn", "--", str(db_path.resolve())],
        ["/usr/sbin/lsof", "-nP", "+L1", "-Fpcfikn"],
    ]

    failed = runtime_ownership.inspect_state_db_ownership(
        db_path,
        runner=lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="lsof: permission denied",
        ),
    )
    assert failed.available is False
    assert failed.errors == ("lsof: permission denied",)


def test_absent_target_uses_unlinked_scan_and_can_prove_quiescent(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "state.db"
    commands: list[list[str]] = []
    monkeypatch.setattr(
        runtime_ownership.shutil, "which", lambda _name: "/usr/sbin/lsof"
    )

    def runner(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=1, stdout="", stderr="")

    report = runtime_ownership.inspect_state_db_ownership(
        db_path,
        runner=runner,
    )

    assert report.available is True
    assert report.pids == ()
    assert commands == [["/usr/sbin/lsof", "-nP", "+L1", "-Fpcfikn"]]


def test_absent_target_still_detects_fully_unlinked_old_generation(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "state.db"
    monkeypatch.setattr(
        runtime_ownership.shutil, "which", lambda _name: "/usr/sbin/lsof"
    )
    output = _lsof_file(
        pid=222,
        command="python",
        fd="8u",
        inode=9001,
        links=0,
        path=str(db_path),
        deleted=True,
    )
    report = runtime_ownership.inspect_state_db_ownership(
        db_path,
        runner=lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=output,
            stderr="",
        ),
    )

    assert report.available is True
    assert report.pids == (222,)
    assert report.unlinked_files


def test_default_dashboard_without_home_override_blocks_darwin_quiescence(
    tmp_path, monkeypatch
):
    import plistlib
    import pwd

    user_home = tmp_path / "user"
    launch_agents = user_home / "Library" / "LaunchAgents"
    launch_agents.mkdir(parents=True)
    plist_path = launch_agents / "ai.hermes.dashboard.local.plist"
    with plist_path.open("wb") as handle:
        plistlib.dump(
            {
                "Label": "ai.hermes.dashboard.local",
                "ProgramArguments": ["/usr/local/bin/hermes", "dashboard"],
            },
            handle,
        )

    monkeypatch.setattr(runtime_ownership.sys, "platform", "darwin")
    monkeypatch.setattr(
        pwd,
        "getpwuid",
        lambda _uid: SimpleNamespace(pw_dir=str(user_home)),
    )
    monkeypatch.setattr(
        runtime_ownership.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="",
            stderr="",
        ),
    )

    labels = runtime_ownership._loaded_managed_service_labels(
        user_home / ".hermes"
    )
    assert labels == ("ai.hermes.dashboard.local",)


def test_inspection_finds_fully_unlinked_generation_without_linked_owner(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "state.db"
    db_path.write_bytes(b"replacement")
    unlinked = _lsof_file(
        pid=201,
        command="python",
        fd="8u",
        inode=9001,
        links=0,
        path=str(db_path),
        deleted=True,
    ) + _lsof_file(
        pid=201,
        command="python",
        fd="9u",
        inode=9002,
        links=0,
        path=f"{db_path}-wal",
        deleted=True,
    )
    results = iter(
        (
            SimpleNamespace(returncode=1, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout=unlinked, stderr=""),
        )
    )
    monkeypatch.setattr(
        runtime_ownership.shutil, "which", lambda _name: "/usr/sbin/lsof"
    )
    monkeypatch.setattr(
        runtime_ownership,
        "_full_process_command",
        lambda pid, fallback: f"{fallback} --pid {pid}",
    )

    report = runtime_ownership.inspect_state_db_ownership(
        db_path,
        runner=lambda *_args, **_kwargs: next(results),
    )

    assert report.available is True
    assert report.pids == (201,)
    assert {item.kind for item in report.unlinked_files} == {"main", "wal"}
    assert report.foreign_inodes == {"main": {9001}}


def test_windows_psutil_fallback_proves_quiescent_and_finds_live_owner(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "state.db"
    db_path.write_bytes(b"db")
    monkeypatch.setattr(runtime_ownership.sys, "platform", "win32")

    quiescent = runtime_ownership.inspect_state_db_ownership(
        db_path,
        process_iter=lambda _attrs: (),
    )
    assert quiescent.available is True
    assert quiescent.files == ()

    class FakeProcess:
        pid = 73
        info = {
            "pid": 73,
            "name": "python.exe",
            "cmdline": ["python.exe", "-m", "hermes_cli.main", "dashboard"],
        }

        @staticmethod
        def open_files():
            return [SimpleNamespace(path=str(db_path), fd=11)]

    live = runtime_ownership.inspect_state_db_ownership(
        db_path,
        process_iter=lambda _attrs: (FakeProcess(),),
    )
    assert live.available is True
    assert live.pids == (73,)
    assert live.files[0].kind == "main"
    assert live.files[0].command.endswith("hermes_cli.main dashboard")


def test_windows_psutil_fallback_filters_other_users_and_fails_on_own_errors(
    tmp_path, monkeypatch
):
    import os
    import psutil

    db_path = tmp_path / "state.db"
    db_path.write_bytes(b"db")
    monkeypatch.setattr(runtime_ownership.sys, "platform", "win32")

    class OtherUserProcess:
        pid = 80
        info = {
            "pid": 80,
            "name": "system.exe",
            "cmdline": [],
            "username": "OTHER-DOMAIN\\someone-else",
        }

        @staticmethod
        def open_files():
            raise AssertionError("other-user process should be filtered")

    filtered = runtime_ownership.inspect_state_db_ownership(
        db_path,
        process_iter=lambda _attrs: (OtherUserProcess(),),
    )
    assert filtered.available is True

    class CurrentUserDeniedProcess:
        pid = 81
        info = {
            "pid": 81,
            "name": "python.exe",
            "cmdline": ["python.exe", "-m", "gateway.run"],
            "username": psutil.Process(os.getpid()).username(),
        }

        @staticmethod
        def open_files():
            raise OSError("access denied")

    denied = runtime_ownership.inspect_state_db_ownership(
        db_path,
        process_iter=lambda _attrs: (CurrentUserDeniedProcess(),),
    )
    assert denied.available is False
    assert denied.errors == ("access denied",)


def test_psutil_fallback_ignores_poisoned_username_environment(
    tmp_path, monkeypatch
):
    import os
    import psutil

    db_path = tmp_path / "state.db"
    db_path.write_bytes(b"db")
    monkeypatch.setattr(runtime_ownership.sys, "platform", "linux")
    monkeypatch.setattr(runtime_ownership.shutil, "which", lambda _name: None)
    monkeypatch.setenv("USER", "poisoned-user")
    monkeypatch.setenv("LOGNAME", "poisoned-user")
    monkeypatch.setenv("USERNAME", "poisoned-user")
    current_username = psutil.Process(os.getpid()).username()

    class CurrentOwner:
        pid = 82
        info = {
            "pid": 82,
            "name": "python",
            "cmdline": ["python", "-m", "gateway.run"],
            "username": current_username,
        }

        @staticmethod
        def open_files():
            return [SimpleNamespace(path=str(db_path), fd=13)]

    report = runtime_ownership.inspect_state_db_ownership(
        db_path,
        process_iter=lambda _attrs: (CurrentOwner(),),
    )

    assert report.available is False
    assert report.pids == (82,)


def test_unix_psutil_fallback_only_proves_stopped_database(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    db_path.write_bytes(b"db")
    monkeypatch.setattr(runtime_ownership.sys, "platform", "linux")
    monkeypatch.setattr(runtime_ownership.shutil, "which", lambda _name: None)

    quiescent = runtime_ownership.inspect_state_db_ownership(
        db_path,
        process_iter=lambda _attrs: (),
    )
    assert quiescent.available is True
    assert quiescent.files == ()

    class FakeProcess:
        pid = 74
        info = {
            "pid": 74,
            "name": "python",
            "cmdline": ["python", "-m", "gateway.run"],
            "username": "",
        }

        @staticmethod
        def open_files():
            return [SimpleNamespace(path=str(db_path), fd=12)]

    live = runtime_ownership.inspect_state_db_ownership(
        db_path,
        process_iter=lambda _attrs: (FakeProcess(),),
    )
    assert live.available is False
    assert live.pids == (74,)
    assert "cannot verify Unix inode generations" in live.errors[-1]


def test_runtime_identity_round_trip_records_boot_source(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_ownership.os, "getpid", lambda: 321)
    monkeypatch.setattr(
        runtime_ownership, "_current_process_start_time", lambda _pid: 654
    )
    monkeypatch.setattr(
        runtime_ownership,
        "_current_source_status",
        lambda: {
            "boot_revision": "abc123",
            "disk_revision": "abc123",
            "code_skew": False,
        },
    )

    path = runtime_ownership.write_runtime_identity(
        "dashboard",
        home=tmp_path,
        details={"port": 9119},
    )

    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["kind"] == "dashboard"
    assert payload["pid"] == 321
    assert payload["start_time"] == 654
    assert payload["source"]["boot_revision"] == "abc123"
    assert payload["details"] == {"port": 9119}

    runtime_ownership.remove_runtime_identity(path)
    assert not path.exists()


def test_runtime_identity_rejects_unverifiable_recorded_start_time(monkeypatch):
    monkeypatch.setattr(
        runtime_ownership,
        "_current_process_start_time",
        lambda _pid: None,
    )

    assert not runtime_ownership._identity_matches_pid(
        {"pid": 55, "start_time": 123},
        55,
    )


def test_owner_summary_flags_stale_dashboard_revision(tmp_path, monkeypatch):
    pid = 77
    identity_dir = tmp_path / "runtime" / "process-identities"
    identity_dir.mkdir(parents=True)
    (identity_dir / f"dashboard-{pid}.json").write_text(
        json.dumps({
            "kind": "dashboard",
            "pid": pid,
            "start_time": 123,
            "source": {
                "boot_revision": "oldrev",
                "disk_revision": "oldrev",
                "code_skew": False,
            },
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runtime_ownership, "_current_process_start_time", lambda _pid: 123
    )
    monkeypatch.setattr(
        runtime_ownership,
        "_current_source_status",
        lambda: {
            "boot_revision": None,
            "disk_revision": "newrev",
            "code_skew": None,
        },
    )
    report = runtime_ownership.StateDbOwnershipReport(
        db_path=tmp_path / "state.db",
        available=True,
        files=(
            runtime_ownership.OpenStateDbFile(
                pid=pid,
                command="python -m hermes_cli.main dashboard",
                fd="9u",
                path=str(tmp_path / "state.db"),
                kind="main",
                inode=100,
                link_count=1,
            ),
        ),
        current_inodes={"main": 100},
    )

    owners = runtime_ownership.summarize_state_db_owners(report, home=tmp_path)

    assert len(owners) == 1
    assert owners[0].role == "dashboard"
    assert owners[0].boot_revision == "oldrev"
    assert owners[0].disk_revision == "newrev"
    assert owners[0].stale_revision is True


def test_named_profile_owner_finds_machine_dashboard_identity(tmp_path, monkeypatch):
    root_home = tmp_path / ".hermes"
    profile_home = root_home / "profiles" / "trading"
    profile_home.mkdir(parents=True)
    pid = 88
    identity_dir = root_home / "runtime" / "process-identities"
    identity_dir.mkdir(parents=True)
    (identity_dir / f"dashboard-{pid}.json").write_text(
        json.dumps(
            {
                "kind": "dashboard",
                "pid": pid,
                "start_time": 456,
                "source": {
                    "boot_revision": "adopted-rev",
                    "disk_revision": "adopted-rev",
                    "code_skew": False,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runtime_ownership, "_current_process_start_time", lambda _pid: 456
    )
    monkeypatch.setattr(
        runtime_ownership,
        "_current_source_status",
        lambda: {
            "boot_revision": None,
            "disk_revision": "adopted-rev",
            "code_skew": None,
        },
    )
    report = runtime_ownership.StateDbOwnershipReport(
        db_path=profile_home / "state.db",
        available=True,
        files=(
            runtime_ownership.OpenStateDbFile(
                pid=pid,
                command="python -m hermes_cli.main dashboard",
                fd="9u",
                path=str(profile_home / "state.db"),
                kind="main",
                inode=111,
                link_count=1,
            ),
        ),
        current_inodes={"main": 111},
    )

    owners = runtime_ownership.summarize_state_db_owners(
        report,
        home=profile_home,
    )

    assert owners[0].role == "dashboard"
    assert owners[0].boot_revision == "adopted-rev"
    assert owners[0].stale_revision is False


def test_doctor_marks_split_runtime_ownership_as_manual_issue(
    tmp_path, monkeypatch, capsys
):
    from hermes_cli import doctor

    db_path = tmp_path / "state.db"
    report = runtime_ownership.StateDbOwnershipReport(
        db_path=db_path,
        available=True,
        files=(
            runtime_ownership.OpenStateDbFile(
                pid=1,
                command="hermes gateway run",
                fd="10u",
                path=f"{db_path}-wal",
                kind="wal",
                inode=100,
                link_count=1,
            ),
            runtime_ownership.OpenStateDbFile(
                pid=2,
                command="hermes dashboard",
                fd="11u",
                path=f"{db_path}-wal",
                kind="wal",
                inode=200,
                link_count=0,
                deleted=True,
            ),
        ),
        current_inodes={"wal": 100},
    )
    owners = (
        runtime_ownership.StateDbOwner(
            pid=1,
            command="hermes gateway run",
            role="gateway",
            boot_revision="newrev",
            disk_revision="newrev",
        ),
        runtime_ownership.StateDbOwner(
            pid=2,
            command="hermes dashboard",
            role="dashboard",
            boot_revision="oldrev",
            disk_revision="newrev",
        ),
    )
    monkeypatch.setattr(
        runtime_ownership, "inspect_state_db_ownership", lambda _path: report
    )
    monkeypatch.setattr(
        runtime_ownership,
        "summarize_state_db_owners",
        lambda _report, *, home: owners,
    )
    issues: list[str] = []

    status = doctor._check_state_db_runtime_ownership(db_path, issues)

    output = capsys.readouterr().out
    assert status == doctor._STATE_DB_OWNERSHIP_UNSAFE
    assert "state.db runtime ownership is unsafe" in output
    assert "split inode generations" in output
    assert "unlinked open sidecars" in output
    assert "oldrev->newrev" in output
    assert len(issues) == 1
    assert "coordinated cohort" in issues[0]


def test_doctor_distinguishes_quiescent_and_coherent_live_ownership(
    tmp_path, monkeypatch
):
    from hermes_cli import doctor

    db_path = tmp_path / "state.db"
    report = runtime_ownership.StateDbOwnershipReport(
        db_path=db_path,
        available=True,
        files=(),
        current_inodes={},
    )
    monkeypatch.setattr(
        runtime_ownership, "inspect_state_db_ownership", lambda _path: report
    )
    monkeypatch.setattr(
        runtime_ownership,
        "summarize_state_db_owners",
        lambda _report, *, home: (),
    )

    status = doctor._check_state_db_runtime_ownership(db_path, [])
    assert status == doctor._STATE_DB_OWNERSHIP_QUIESCENT

    owner = runtime_ownership.StateDbOwner(
        pid=99,
        command="hermes dashboard",
        role="dashboard",
        boot_revision="same-rev",
        disk_revision="same-rev",
    )
    monkeypatch.setattr(
        runtime_ownership,
        "summarize_state_db_owners",
        lambda _report, *, home: (owner,),
    )

    status = doctor._check_state_db_runtime_ownership(db_path, [])
    assert status == doctor._STATE_DB_OWNERSHIP_LIVE


def test_doctor_unknown_ownership_is_a_manual_health_issue(
    tmp_path, monkeypatch
):
    from hermes_cli import doctor

    db_path = tmp_path / "state.db"
    report = runtime_ownership.StateDbOwnershipReport(
        db_path=db_path,
        available=False,
        files=(),
        current_inodes={},
        errors=("lsof permission denied",),
    )
    monkeypatch.setattr(
        runtime_ownership, "inspect_state_db_ownership", lambda _path: report
    )
    issues: list[str] = []

    status = doctor._check_state_db_runtime_ownership(db_path, issues)

    assert status == doctor._STATE_DB_OWNERSHIP_UNKNOWN
    assert len(issues) == 1
    assert "before treating persistence as healthy" in issues[0]


def test_doctor_unknown_ownership_cannot_end_all_passed(
    tmp_path, monkeypatch
):
    from argparse import Namespace
    import contextlib
    import io
    import sys

    from hermes_cli import doctor

    home = tmp_path / ".hermes"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    (home / "state.db").write_bytes(b"not-opened")
    (home / "config.yaml").write_text("memory: {}\n", encoding="utf-8")
    monkeypatch.setattr(doctor, "HERMES_HOME", home)
    monkeypatch.setattr(doctor, "PROJECT_ROOT", project)
    monkeypatch.setattr(doctor, "_DHH", str(home))
    monkeypatch.setenv("HERMES_HOME", str(home))

    report = runtime_ownership.StateDbOwnershipReport(
        db_path=home / "state.db",
        available=False,
        files=(),
        current_inodes={},
        errors=("lsof permission denied",),
    )
    monkeypatch.setattr(
        runtime_ownership, "inspect_state_db_ownership", lambda _path: report
    )
    monkeypatch.setitem(
        sys.modules,
        "model_tools",
        SimpleNamespace(
            check_tool_availability=lambda *a, **kw: ([], []),
            TOOLSET_REQUIREMENTS={},
        ),
    )

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        doctor.run_doctor(Namespace(fix=False, ack=None))

    rendered = output.getvalue()
    assert "State.db ownership could not be inspected" in rendered
    assert "All checks passed!" not in rendered


def test_doctor_coherent_live_owner_defers_green_write_health_verdict(
    tmp_path, monkeypatch
):
    from argparse import Namespace
    import contextlib
    import io
    import sqlite3
    import sys

    from hermes_cli import doctor

    home = tmp_path / ".hermes"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    db_path = home / "state.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")
        conn.commit()
    finally:
        conn.close()
    (home / "config.yaml").write_text("memory: {}\n", encoding="utf-8")
    monkeypatch.setattr(doctor, "HERMES_HOME", home)
    monkeypatch.setattr(doctor, "PROJECT_ROOT", project)
    monkeypatch.setattr(doctor, "_DHH", str(home))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(
        doctor,
        "_check_state_db_runtime_ownership",
        lambda _path, _issues: doctor._STATE_DB_OWNERSHIP_LIVE,
    )
    monkeypatch.setitem(
        sys.modules,
        "model_tools",
        SimpleNamespace(
            check_tool_availability=lambda *a, **kw: ([], []),
            TOOLSET_REQUIREMENTS={},
        ),
    )

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        doctor.run_doctor(Namespace(fix=False, ack=None))

    rendered = output.getvalue()
    assert "State.db write health was not verified" in rendered
    assert "All checks passed!" not in rendered


def test_doctor_state_db_mutations_require_proven_quiescence(capsys):
    from hermes_cli import doctor

    issues: list[str] = []
    assert doctor._state_db_mutation_allowed(
        doctor._STATE_DB_OWNERSHIP_QUIESCENT,
        "state.db repair",
        issues,
    )
    assert issues == []

    for status in (
        doctor._STATE_DB_OWNERSHIP_LIVE,
        doctor._STATE_DB_OWNERSHIP_UNSAFE,
        doctor._STATE_DB_OWNERSHIP_UNKNOWN,
    ):
        assert not doctor._state_db_mutation_allowed(
            status,
            "state.db repair",
            issues,
        )

    output = capsys.readouterr().out
    assert output.count("Skipped state.db repair") == 3
    assert len(issues) == 1
    assert "confirm state.db has no live owners" in issues[0]


def test_doctor_rechecks_quiescence_immediately_before_mutation(
    tmp_path, monkeypatch
):
    from hermes_cli import doctor

    db_path = tmp_path / "state.db"
    report = runtime_ownership.StateDbOwnershipReport(
        db_path=db_path,
        available=True,
        files=(
            runtime_ownership.OpenStateDbFile(
                pid=101,
                command="hermes dashboard",
                fd="7u",
                path=str(db_path),
                kind="main",
                inode=123,
                link_count=1,
            ),
        ),
        current_inodes={"main": 123},
    )
    monkeypatch.setattr(
        runtime_ownership, "inspect_state_db_ownership", lambda _path: report
    )
    issues: list[str] = []

    allowed = doctor._state_db_mutation_allowed(
        doctor._STATE_DB_OWNERSHIP_QUIESCENT,
        "state.db repair",
        issues,
        state_db_path=db_path,
    )

    assert allowed is False
    assert len(issues) == 1


def test_doctor_closes_failed_select_before_quiescent_schema_repair(
    tmp_path, monkeypatch
):
    """Malformed SELECT must release Doctor's own FD before the repair recheck."""
    from argparse import Namespace
    import sqlite3

    import hermes_state
    from hermes_cli import doctor

    home = tmp_path / ".hermes"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    db_path = home / "state.db"
    db_path.write_bytes(b"malformed-placeholder")

    monkeypatch.setattr(doctor, "HERMES_HOME", home)
    monkeypatch.setattr(doctor, "PROJECT_ROOT", project)
    monkeypatch.setattr(doctor, "_DHH", str(home))
    monkeypatch.setenv("HERMES_HOME", str(home))

    events: list[str] = []
    real_connect = sqlite3.connect
    state_connect_count = 0

    class FailedSelectConnection:
        closed = False

        def execute(self, _sql):
            events.append("malformed-select")
            raise sqlite3.DatabaseError("database disk image is malformed")

        def close(self):
            self.closed = True
            events.append("failed-select-closed")

    failed_connection = FailedSelectConnection()

    class CountConnection:
        def execute(self, _sql):
            events.append("post-repair-select")
            return SimpleNamespace(fetchone=lambda: (7,))

        def close(self):
            events.append("post-repair-closed")

    def guarded_connect(database, *args, **kwargs):
        nonlocal state_connect_count
        if str(database) != str(db_path):
            return real_connect(database, *args, **kwargs)
        state_connect_count += 1
        if state_connect_count == 1:
            return failed_connection
        return CountConnection()

    def initial_ownership(_path, _issues):
        events.append("initial-ownership")
        return doctor._STATE_DB_OWNERSHIP_QUIESCENT

    def repair_recheck(_path):
        assert failed_connection.closed is True
        events.append("repair-recheck")
        return runtime_ownership.StateDbOwnershipReport(
            db_path=db_path,
            available=True,
            files=(),
            current_inodes={"main": 1},
        )

    def repair(_path):
        events.append("repair")
        return {
            "repaired": True,
            "strategy": "test",
            "backup_path": None,
        }

    monkeypatch.setattr(sqlite3, "connect", guarded_connect)
    monkeypatch.setattr(
        doctor,
        "_check_state_db_runtime_ownership",
        initial_ownership,
    )
    monkeypatch.setattr(
        runtime_ownership,
        "inspect_state_db_ownership",
        repair_recheck,
    )
    monkeypatch.setattr(hermes_state, "repair_state_db_schema", repair)
    monkeypatch.setattr(
        doctor,
        "_check_gateway_service_linger",
        lambda _issues: (_ for _ in ()).throw(SystemExit(0)),
    )

    with pytest.raises(SystemExit):
        doctor.run_doctor(Namespace(fix=True, ack=None))

    assert events.index("failed-select-closed") < events.index("repair-recheck")
    assert events.index("repair-recheck") < events.index("repair")
    assert "post-repair-select" in events


@pytest.mark.parametrize("ownership_status", ["unsafe", "unknown"])
def test_doctor_probes_ownership_before_db_access_and_blocks_unproven_fix(
    tmp_path, monkeypatch, capsys, ownership_status
):
    """Unsafe or unknown ownership must prevent DB opens and WAL checkpoints."""
    from argparse import Namespace
    import sqlite3

    from hermes_cli import doctor

    home = tmp_path / ".hermes"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    (home / "state.db").write_bytes(b"not-opened")
    with Path(f"{home / 'state.db'}-wal").open("wb") as wal_file:
        wal_file.truncate(51 * 1024 * 1024)

    monkeypatch.setattr(doctor, "HERMES_HOME", home)
    monkeypatch.setattr(doctor, "PROJECT_ROOT", project)
    monkeypatch.setattr(doctor, "_DHH", str(home))
    monkeypatch.setenv("HERMES_HOME", str(home))

    events: list[str] = []
    real_connect = sqlite3.connect

    def ownership_probe(_path, _issues):
        events.append("ownership")
        return ownership_status

    def guarded_connect(database, *args, **kwargs):
        if str(database) == str(home / "state.db"):
            events.append("connect")
            raise AssertionError("unsafe state.db must not be opened")
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(doctor, "_check_state_db_runtime_ownership", ownership_probe)
    monkeypatch.setattr(sqlite3, "connect", guarded_connect)
    monkeypatch.setattr(
        doctor,
        "_check_gateway_service_linger",
        lambda _issues: (_ for _ in ()).throw(SystemExit(0)),
    )

    with pytest.raises(SystemExit):
        doctor.run_doctor(Namespace(fix=True, ack=None))

    output = capsys.readouterr().out
    assert events == ["ownership"]
    assert "Skipped state.db content probes" in output
    assert "Skipped state.db WAL checkpoint" in output
