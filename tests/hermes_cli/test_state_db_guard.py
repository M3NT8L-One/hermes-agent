"""Tests for the state.db integrity guard used by the update flow (#68474).

Exercises ``verify_sqlite_integrity`` and ``copy_db_and_verify`` against REAL
SQLite files (valid, zeroed, truncated) — the exact corruption signature from
issue #68474 (file kept at original size, 100% null bytes, header gone).
"""

import sqlite3
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.backup import (
    copy_db_and_verify,
    replace_sqlite_db_cohort_offline,
    replace_sqlite_db_offline,
    verify_sqlite_integrity,
)


@pytest.fixture()
def valid_db(tmp_path):
    path = tmp_path / "state.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY, name TEXT)")
    conn.executemany(
        "INSERT INTO sessions (name) VALUES (?)", [(f"s{i}",) for i in range(50)]
    )
    conn.commit()
    conn.close()
    return path


def test_valid_db_passes(valid_db):
    res = verify_sqlite_integrity(valid_db)
    assert res["valid"] is True
    assert res["size"] == valid_db.stat().st_size
    assert "passed" in res["message"]


def test_zeroed_db_fails_header_check(valid_db):
    # The #68474 signature: same size, all null bytes.
    size = valid_db.stat().st_size
    valid_db.write_bytes(b"\x00" * size)
    res = verify_sqlite_integrity(valid_db)
    assert res["valid"] is False
    assert "header" in res["message"]


def test_missing_file():
    from pathlib import Path

    res = verify_sqlite_integrity(Path("/nonexistent/state.db"))
    assert res["valid"] is False
    assert "not found" in res["message"]


def test_too_small_file(tmp_path):
    path = tmp_path / "state.db"
    path.write_bytes(b"SQLite")
    res = verify_sqlite_integrity(path)
    assert res["valid"] is False
    assert "too small" in res["message"]


def test_header_ok_but_garbage_body_fails_pragma(tmp_path):
    path = tmp_path / "state.db"
    path.write_bytes(b"SQLite format 3\0" + b"\xff" * 4096)
    res = verify_sqlite_integrity(path)
    assert res["valid"] is False


def test_oversized_db_skips_pragma_but_still_checks_header(valid_db):
    res = verify_sqlite_integrity(valid_db, max_bytes=1)
    # Header intact + schema probe passes → pass without the full pragma.
    assert res["valid"] is True
    assert "skipped PRAGMA integrity_check" in res["message"]
    size = valid_db.stat().st_size
    valid_db.write_bytes(b"\x00" * size)
    res = verify_sqlite_integrity(valid_db, max_bytes=1)
    # Zeroed header must still fail even when pragma is skipped for size.
    assert res["valid"] is False


def test_default_max_bytes_bounds_the_pragma_by_size():
    """The default must NOT be size-unbounded.

    ``PRAGMA integrity_check`` walks every page in the file, so an unbounded
    default made `hermes update` peg a CPU for minutes on a multi-GB
    state.db with no output (read as a hang). Callers that omit max_bytes
    must inherit a finite ceiling.
    """
    import inspect

    from hermes_cli.backup import DEFAULT_INTEGRITY_CHECK_MAX_BYTES

    default = inspect.signature(verify_sqlite_integrity).parameters["max_bytes"].default
    assert default == DEFAULT_INTEGRITY_CHECK_MAX_BYTES
    assert default > 0, "size-unbounded integrity_check is never a safe default"


def test_oversized_db_probe_catches_malformed_schema(tmp_path):
    """Skipping the pragma must not mean skipping corruption detection.

    A file with a valid header whose schema cannot be parsed has to fail
    the oversized path via the cheap structural probe.
    """
    path = tmp_path / "state.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (a INTEGER)")
    conn.commit()
    conn.close()

    raw = bytearray(path.read_bytes())
    # Corrupt the schema b-tree page (page 2 onward) while leaving the
    # 16-byte header magic intact, so only the probe can catch it.
    for i in range(100, min(len(raw), 4096)):
        raw[i] = 0xFF
    path.write_bytes(bytes(raw))

    res = verify_sqlite_integrity(path, max_bytes=1)
    assert res["valid"] is False
    assert "probe" in res["message"]


def test_max_bytes_zero_forces_full_check(valid_db):
    """``max_bytes=0`` remains the explicit opt-in for a full scan."""
    res = verify_sqlite_integrity(valid_db, max_bytes=0)
    assert res["valid"] is True
    assert "integrity check passed" in res["message"]


def test_copy_db_and_verify_roundtrip(valid_db, tmp_path):
    dst = tmp_path / "snapshot" / "state.db"
    dst.parent.mkdir()
    assert copy_db_and_verify(valid_db, dst) is True
    assert verify_sqlite_integrity(dst)["valid"] is True


def test_copy_db_and_verify_refuses_zeroed_source(valid_db, tmp_path):
    size = valid_db.stat().st_size
    valid_db.write_bytes(b"\x00" * size)
    dst = tmp_path / "snapshot" / "state.db"
    dst.parent.mkdir()
    assert copy_db_and_verify(valid_db, dst) is False
    assert not dst.exists()


def test_restore_flow_end_to_end(valid_db, tmp_path):
    """Simulate the #68474 recovery path: live db zeroed, snapshot valid →
    restore snapshot over live file → verify restored copy."""
    import shutil

    snap = tmp_path / "state-snapshots" / "20260721-pre-update" / "state.db"
    snap.parent.mkdir(parents=True)
    shutil.copy2(valid_db, snap)

    # Zero the live db (the bug).
    size = valid_db.stat().st_size
    valid_db.write_bytes(b"\x00" * size)
    assert verify_sqlite_integrity(valid_db)["valid"] is False
    assert verify_sqlite_integrity(snap)["valid"] is True

    # The guard's restore step.
    shutil.copy2(snap, valid_db)
    restored = verify_sqlite_integrity(valid_db)
    assert restored["valid"] is True
    conn = sqlite3.connect(valid_db)
    assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 50
    conn.close()


def _quiescent(_path):
    return SimpleNamespace(quiescent=True, reason="test cohort offline")


def _cohort_hashes(path: Path) -> dict[str, str]:
    result = {}
    for suffix in ("", "-wal", "-shm", "-journal"):
        member = Path(f"{path}{suffix}")
        if member.exists():
            result[suffix or "main"] = hashlib.sha256(
                member.read_bytes()
            ).hexdigest()
    return result


def _value_db(path: Path, value: str) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE value_store (value TEXT)")
        conn.execute("INSERT INTO value_store VALUES (?)", (value,))


def test_offline_replace_normalizes_wal_and_preserves_source_exactly(tmp_path):
    source = tmp_path / "source.db"
    writer = sqlite3.connect(source)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.execute("CREATE TABLE value_store (value TEXT)")
    writer.execute("INSERT INTO value_store VALUES ('committed-in-wal')")
    writer.commit()
    before = _cohort_hashes(source)
    assert "-wal" in before

    target = tmp_path / "target.db"
    _value_db(target, "old")
    result = replace_sqlite_db_offline(
        source,
        target,
        reason="wal normalization regression",
        quiescence_fn=_quiescent,
    )
    try:
        assert result["applied"] is True, result
        with sqlite3.connect(target) as conn:
            assert conn.execute("SELECT value FROM value_store").fetchall() == [
                ("committed-in-wal",)
            ]
        assert _cohort_hashes(source) == before
        assert not list(tmp_path.glob(".*.staged-*"))
    finally:
        writer.close()


def test_offline_replace_rolls_back_exact_cohort_on_second_move_failure(
    tmp_path, monkeypatch
):
    from hermes_cli import backup

    source = tmp_path / "source.db"
    _value_db(source, "new")
    target = tmp_path / "target.db"
    _value_db(target, "old")
    Path(f"{target}-wal").write_bytes(b"old-wal")
    Path(f"{target}-shm").write_bytes(b"old-shm")
    before = {
        suffix: Path(f"{target}{suffix}").read_bytes()
        for suffix in ("", "-wal", "-shm")
    }
    real_replace = backup.os.replace
    failed = False

    def fail_second_member(src, dst):
        nonlocal failed
        if Path(src) == Path(f"{target}-wal") and not failed:
            failed = True
            raise OSError("injected WAL move failure")
        return real_replace(src, dst)

    monkeypatch.setattr(backup.os, "replace", fail_second_member)
    result = replace_sqlite_db_offline(
        source,
        target,
        reason="move failure regression",
        verify_fn=lambda _path: {"valid": True, "message": "ok"},
        quiescence_fn=_quiescent,
    )

    assert result["applied"] is False
    assert result["rollback_performed"] is True
    assert {
        suffix: Path(f"{target}{suffix}").read_bytes()
        for suffix in ("", "-wal", "-shm")
    } == before


def test_installed_verification_failure_rolls_back_original(tmp_path):
    source = tmp_path / "source.db"
    _value_db(source, "new")
    target = tmp_path / "target.db"
    _value_db(target, "old")

    def verifier(path):
        if Path(path) == target:
            return {"valid": False, "message": "injected installed failure"}
        return {"valid": True, "message": "ok"}

    result = replace_sqlite_db_offline(
        source,
        target,
        reason="installed verification regression",
        verify_fn=verifier,
        quiescence_fn=_quiescent,
    )

    assert result["rollback_performed"] is True
    with sqlite3.connect(target) as conn:
        assert conn.execute("SELECT value FROM value_store").fetchone() == (
            "old",
        )
    receipt = json.loads(Path(result["receipt_path"]).read_text())
    assert receipt["status"] == "rolled-back"


def test_owner_appearance_before_install_requires_manual_recovery(tmp_path):
    source = tmp_path / "source.db"
    _value_db(source, "new")
    target = tmp_path / "target.db"
    _value_db(target, "old")
    target_calls = 0

    def ownership(path):
        nonlocal target_calls
        if Path(path) == target:
            target_calls += 1
            offline = target_calls <= 2
            return SimpleNamespace(
                quiescent=offline,
                reason="offline" if offline else "owner appeared",
            )
        return _quiescent(path)

    result = replace_sqlite_db_offline(
        source,
        target,
        reason="owner appearance regression",
        verify_fn=lambda _path: {"valid": True, "message": "ok"},
        quiescence_fn=ownership,
    )

    assert result["applied"] is False
    assert result["rollback_performed"] is False
    receipt = json.loads(Path(result["receipt_path"]).read_text())
    assert receipt["status"] == "manual-recovery-required"
    assert Path(result["evidence_dir"], "original-target.db").exists()


def test_replacement_integrity_forces_full_check(tmp_path, monkeypatch):
    from hermes_cli import backup

    path = tmp_path / "generic.db"
    _value_db(path, "x")
    calls = []

    def verifier(_path, **kwargs):
        calls.append(kwargs)
        return {"valid": True, "message": "ok"}

    monkeypatch.setattr(backup, "verify_sqlite_integrity", verifier)
    result = backup._replacement_integrity_check(
        path,
        tmp_path / "generic-target.db",
        verify_fn=None,
    )
    assert result["valid"] is True
    assert calls == [{"max_bytes": 0}]


def test_cohort_service_guard_refuses_non_state_db_before_mutation(tmp_path):
    source = tmp_path / "source-kanban.db"
    _value_db(source, "new")
    target = tmp_path / "kanban.db"
    _value_db(target, "old")
    guard = tmp_path / "state.db"

    def ownership(path):
        if Path(path) == guard:
            return SimpleNamespace(
                quiescent=False,
                reason="gateway launchd service loaded",
            )
        return _quiescent(path)

    result = replace_sqlite_db_cohort_offline(
        [(source, target)],
        reason="service guard regression",
        evidence_root=tmp_path / "evidence",
        service_guard_paths=[guard],
        verify_fn=lambda _path: {"valid": True, "message": "ok"},
        quiescence_fn=ownership,
    )

    assert result["deferred"] is True
    with sqlite3.connect(target) as conn:
        assert conn.execute("SELECT value FROM value_store").fetchone() == (
            "old",
        )


def test_offline_replace_installs_exact_candidate_when_target_is_absent(tmp_path):
    source = tmp_path / "source.db"
    _value_db(source, "recovered")
    target = tmp_path / "missing.db"

    result = replace_sqlite_db_offline(
        source,
        target,
        reason="missing target recovery",
        quiescence_fn=_quiescent,
    )

    assert result["applied"] is True
    assert (target.stat().st_mode & 0o777) == 0o600
    with sqlite3.connect(target) as conn:
        assert conn.execute("SELECT value FROM value_store").fetchone() == (
            "recovered",
        )


class TestPreUpdateBackupIntegrityGuard:
    """E2E: run the real ``_run_pre_update_backup`` against a temp
    HERMES_HOME whose state.db is corrupted mid-flight (#68474)."""

    @pytest.fixture()
    def hermes_home(self, tmp_path, monkeypatch):
        from pathlib import Path
        import sys

        root = tmp_path / ".hermes"
        root.mkdir()
        (root / "config.yaml").write_text("model:\n  provider: openrouter\n")
        db = root / "state.db"
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()
        monkeypatch.setenv("HERMES_HOME", str(root))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        for mod in list(sys.modules.keys()):
            if mod.startswith("hermes_cli.config") or mod == "hermes_constants":
                del sys.modules[mod]
        return root

    def test_healthy_db_stays_quiet(self, hermes_home, capsys):
        from argparse import Namespace

        from hermes_cli.main import _run_pre_update_backup

        snap_id = _run_pre_update_backup(Namespace(no_backup=False, backup=False))
        out = capsys.readouterr().out
        assert snap_id is not None
        assert "Pre-update snapshot" in out
        assert "integrity check FAILED" not in out

    def test_zeroed_db_after_snapshot_is_loud(self, hermes_home, capsys, monkeypatch):
        """If state.db is zeroed right after the snapshot completes, the
        guard must warn loudly instead of proceeding silently (exit-0 mask)."""
        from argparse import Namespace

        import hermes_cli.backup as backup_mod
        from hermes_cli.main import _run_pre_update_backup

        real_create = backup_mod.create_quick_snapshot

        def create_then_zero(**kwargs):
            snap_id = real_create(**kwargs)
            live = hermes_home / "state.db"
            live.write_bytes(b"\x00" * live.stat().st_size)
            return snap_id

        monkeypatch.setattr(backup_mod, "create_quick_snapshot", create_then_zero)
        snap_id = _run_pre_update_backup(Namespace(no_backup=False, backup=False))
        out = capsys.readouterr().out
        assert snap_id is not None
        assert "integrity check FAILED" in out
        assert "Snapshot copy is valid" in out
