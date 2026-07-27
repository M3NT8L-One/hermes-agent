"""#68474 hardening: zeroed state.db detection + quarantine."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_zeroed_quarantine_fails_closed_and_preserves_exact_cohort(tmp_path):
    import hermes_state as hs

    db = tmp_path / "state.db"
    db.write_bytes(bytes(1024))
    Path(f"{db}-wal").write_bytes(b"wal-evidence")
    Path(f"{db}-shm").write_bytes(b"shm-evidence")
    Path(f"{db}-journal").write_bytes(b"journal-evidence")
    assert hs.is_zeroed_state_db(db) is True

    q = hs.quarantine_zeroed_state_db(db)
    assert q is None
    assert db.read_bytes() == bytes(1024)
    assert Path(f"{db}-wal").read_bytes() == b"wal-evidence"
    assert Path(f"{db}-shm").read_bytes() == b"shm-evidence"
    assert Path(f"{db}-journal").read_bytes() == b"journal-evidence"


def test_sessiondb_refuses_fresh_database_after_zeroed_detection(
    tmp_path, monkeypatch
):
    import hermes_state as hs
    import sqlite3

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = tmp_path / "state.db"
    db.write_bytes(bytes(4096))

    with pytest.raises(sqlite3.DatabaseError, match="Preserved in place"):
        hs.SessionDB(db_path=db)
    assert db.read_bytes() == bytes(4096)
    assert not list(tmp_path.glob("state.db.zeroed-*.bak"))


def test_concurrent_zeroed_opens_both_fail_closed_without_clobber(tmp_path):
    """#68805: two concurrent startups must not race on quarantine.

    Without the cross-process lock, the second process could move its
    newly-created empty DB over the first process's quarantine backup,
    erasing the original damaged-file evidence. With the lock, the
    second process re-checks under the lock, finds the file no longer
    zeroed (or gone), and returns without clobbering.
    """
    import hermes_state as hs
    import threading

    db = tmp_path / "state.db"
    db.write_bytes(bytes(4096))  # zeroed (all-NUL) 4 KB file

    results: list = [None, None]
    errors: list = [None, None]

    def worker(idx):
        try:
            sdb = hs.SessionDB(db_path=db)
            try:
                results[idx] = "ok"
            finally:
                sdb.close()
        except Exception as exc:
            errors[idx] = exc

    t1 = threading.Thread(target=worker, args=(0,))
    t2 = threading.Thread(target=worker, args=(1,))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert all(error is not None for error in errors)
    assert results == [None, None]
    assert db.read_bytes() == bytes(4096)
    assert not list(tmp_path.glob("state.db.zeroed-*.bak"))


def test_quarantine_fails_closed_when_lock_held(tmp_path):
    """#68805 review: when the cross-process lock cannot be acquired within
    the timeout, quarantine must FAIL CLOSED — return None without moving
    the file. A fail-open fallback would let a slow/paused startup that
    still owns the lock race with the fallback's re-check + rename.
    """
    import hermes_state as hs
    import platform
    import threading

    db = tmp_path / "state.db"
    db.write_bytes(bytes(4096))  # zeroed (all-NUL) 4 KB file

    lock_path = db.with_name(db.name + ".quarantine.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    # Hold the cross-process lock from a background thread so the main
    # thread's quarantine attempt cannot acquire it.
    lock_held = threading.Event()
    release_lock = threading.Event()

    def hold_lock():
        handle = lock_path.open("a+b")
        try:
            if platform.system() == "Windows":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_held.set()
            release_lock.wait(timeout=15)
            if platform.system() == "Windows":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            lock_held.clear()
        finally:
            handle.close()

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert lock_held.wait(timeout=5), "Background thread failed to acquire lock"

    # Reduce the quarantine lock timeout to keep the test fast. We patch
    # the deadline by calling quarantine directly — it uses a 5s timeout,
    # but we only need to verify it returns None without moving the file.
    result = hs.quarantine_zeroed_state_db(db)

    # Must fail closed: return None without moving the zeroed file
    assert result is None, (
        f"quarantine_zeroed_state_db returned {result} — expected None "
        f"(fail-closed when lock is held)"
    )
    assert db.exists(), "Zeroed state.db was moved despite lock being held"
    assert hs.is_zeroed_state_db(db), "File should still be zeroed (not moved)"

    # Release the lock so the background thread can exit cleanly
    release_lock.set()
    holder.join(timeout=5)
