"""Runtime identity and SQLite ownership diagnostics.

Long-lived Hermes processes can share ``state.db``.  That is safe only while
every process participates in the same linked WAL/SHM generation.  This module
keeps the diagnostic mechanics out of ``doctor.py``:

* dashboard/serve processes publish a small, PID-scoped boot identity;
* Doctor parses ``lsof`` field output without depending on human formatting,
  with a fail-closed ``psutil`` quiescence fallback when ``lsof`` is absent;
* live DB owners are correlated with boot revisions when identity metadata is
  available.

All inspection is best-effort and read-only.  Failure to run ``lsof`` returns
an unavailable report instead of guessing that ownership is healthy.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Callable, Iterable

from utils import atomic_json_write


_IDENTITY_DIR = Path("runtime") / "process-identities"
_DB_KINDS = ("main", "wal", "shm", "journal")


@dataclass(frozen=True)
class OpenStateDbFile:
    """One state-database file descriptor reported by ``lsof``."""

    pid: int
    command: str
    fd: str
    path: str
    kind: str
    inode: int | None
    link_count: int | None
    deleted: bool = False


@dataclass(frozen=True)
class StateDbOwner:
    """Process-level view of a live ``state.db`` owner."""

    pid: int
    command: str
    role: str
    boot_revision: str | None
    disk_revision: str | None

    @property
    def stale_revision(self) -> bool:
        return bool(
            self.boot_revision
            and self.disk_revision
            and self.boot_revision != self.disk_revision
        )


@dataclass(frozen=True)
class StateDbOwnershipReport:
    """Result of a best-effort live ownership inspection."""

    db_path: Path
    available: bool
    files: tuple[OpenStateDbFile, ...]
    current_inodes: dict[str, int]
    errors: tuple[str, ...] = ()

    def open_inodes(self, kind: str) -> set[int]:
        return {
            item.inode
            for item in self.files
            if item.kind == kind and item.inode is not None
        }

    @property
    def split_kinds(self) -> dict[str, set[int]]:
        """Kinds with multiple simultaneously-open inode generations."""
        return {
            kind: inodes
            for kind in _DB_KINDS
            if len(inodes := self.open_inodes(kind)) > 1
        }

    @property
    def foreign_inodes(self) -> dict[str, set[int]]:
        """Open inode generations that do not match the linked path."""
        result: dict[str, set[int]] = {}
        for kind in _DB_KINDS:
            current = self.current_inodes.get(kind)
            if current is None:
                continue
            foreign = self.open_inodes(kind) - {current}
            if foreign:
                result[kind] = foreign
        return result

    @property
    def unlinked_files(self) -> tuple[OpenStateDbFile, ...]:
        return tuple(
            item for item in self.files if item.deleted or item.link_count == 0
        )

    @property
    def pids(self) -> tuple[int, ...]:
        return tuple(sorted({item.pid for item in self.files}))


@dataclass(frozen=True)
class StateDbQuiescence:
    """Fail-closed proof that a SQLite main/WAL/SHM cohort has no owners."""

    db_path: Path
    quiescent: bool
    reason: str
    report: StateDbOwnershipReport | None


@dataclass
class _ParsedLsofFile:
    pid: int
    command: str
    fd: str
    path: str
    inode: int | None
    link_count: int | None
    deleted: bool


def _parse_optional_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_lsof_field_output(output: str) -> list[_ParsedLsofFile]:
    """Parse newline-delimited ``lsof -Fpcfikn`` output.

    The field protocol is stable across macOS/Linux and avoids scraping
    column widths.  Repeated file descriptors for one process are preserved;
    callers decide how to aggregate them.
    """

    parsed: list[_ParsedLsofFile] = []
    current_pid: int | None = None
    current_command = ""
    current_file: dict[str, Any] | None = None

    def _flush_file() -> None:
        nonlocal current_file
        if current_pid is None or current_file is None:
            current_file = None
            return
        path = str(current_file.get("path") or "")
        if path:
            parsed.append(
                _ParsedLsofFile(
                    pid=current_pid,
                    command=current_command,
                    fd=str(current_file.get("fd") or ""),
                    path=path,
                    inode=current_file.get("inode"),
                    link_count=current_file.get("link_count"),
                    deleted=bool(current_file.get("deleted")),
                )
            )
        current_file = None

    for raw_line in (output or "").splitlines():
        if not raw_line:
            continue
        field, value = raw_line[0], raw_line[1:]
        if field == "p":
            _flush_file()
            current_pid = _parse_optional_int(value)
            current_command = ""
        elif field == "c":
            current_command = value
        elif field == "f":
            _flush_file()
            current_file = {"fd": value}
        elif current_file is not None and field == "i":
            current_file["inode"] = _parse_optional_int(value)
        elif current_file is not None and field == "k":
            current_file["link_count"] = _parse_optional_int(value)
        elif current_file is not None and field == "n":
            deleted = value.endswith(" (deleted)")
            current_file["deleted"] = deleted
            current_file["path"] = value[: -len(" (deleted)")] if deleted else value
    _flush_file()
    return parsed


def _kind_for_path(path: str, db_path: Path) -> str | None:
    candidate = os.path.normcase(os.path.abspath(os.path.expanduser(path)))
    base = os.path.normcase(str(db_path.expanduser().resolve(strict=False)))
    if candidate == base:
        return "main"
    if candidate == f"{base}-wal":
        return "wal"
    if candidate == f"{base}-shm":
        return "shm"
    if candidate == f"{base}-journal":
        return "journal"
    return None


def _current_db_inodes(db_path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    paths = {
        "main": db_path,
        "wal": Path(f"{db_path}-wal"),
        "shm": Path(f"{db_path}-shm"),
        "journal": Path(f"{db_path}-journal"),
    }
    for kind, path in paths.items():
        try:
            result[kind] = int(path.stat().st_ino)
        except OSError:
            continue
    return result


def _full_process_command(pid: int, fallback: str) -> str:
    try:
        import psutil  # type: ignore

        command = " ".join(psutil.Process(pid).cmdline()).strip()
        return command or fallback
    except Exception:
        return fallback


def _inspect_psutil_state_db_ownership(
    db_path: Path,
    current_inodes: dict[str, int],
    *,
    process_iter: Callable[..., Iterable[Any]] | None = None,
    windows: bool,
) -> StateDbOwnershipReport:
    """Use process handles to prove quiescence when ``lsof`` is unavailable.

    Windows SQLite handles do not permit the Unix stale-generation failure
    mode in normal operation, so linked-file ownership is a sufficient live
    report there. On Unix, psutil cannot reliably distinguish a deleted inode
    generation from a replacement at the same path. It can still prove a
    *stopped* database is quiescent: zero current-user processes have any
    database member open. Any Unix owner or incomplete same-user scan therefore
    returns ``available=False`` and keeps Doctor fail-closed.
    """
    try:
        import psutil  # type: ignore
    except Exception as exc:
        return StateDbOwnershipReport(
            db_path=db_path,
            available=False,
            files=(),
            current_inodes=current_inodes,
            errors=(f"psutil is unavailable: {exc}",),
        )

    iterator = process_iter or psutil.process_iter
    files: list[OpenStateDbFile] = []
    errors: list[str] = []
    seen: set[tuple[int, str, str]] = set()
    try:
        current_username = psutil.Process(os.getpid()).username()
    except Exception:
        # Unknown means inspect every account rather than trusting mutable
        # USER/LOGNAME/USERNAME environment variables or skipping an owner.
        current_username = ""

    def _account_key(username: str) -> str:
        value = (username or "").strip().lower()
        if "\\" in value:
            value = value.rsplit("\\", 1)[-1]
        if "@" in value:
            value = value.split("@", 1)[0]
        return value

    current_account = _account_key(current_username)
    try:
        processes = iterator(["pid", "name", "cmdline", "username"])
        for process in processes:
            info = getattr(process, "info", {}) or {}
            username = str(info.get("username") or "")
            process_account = _account_key(username)
            same_user = not process_account or not current_account
            if process_account and current_account:
                same_user = process_account == current_account
            # Another account cannot normally open a DB inside this user's
            # Hermes home. Filtering it before open_files avoids expected
            # AccessDenied noise without weakening current/unknown-user scans.
            if not same_user:
                continue
            try:
                pid = int(info.get("pid", getattr(process, "pid", -1)))
                cmdline = info.get("cmdline") or ()
                command = " ".join(str(part) for part in cmdline).strip()
                if not command:
                    command = str(info.get("name") or f"PID {pid}")
                for opened in process.open_files():
                    path = str(getattr(opened, "path", "") or "")
                    deleted = path.endswith(" (deleted)")
                    normalized_path = (
                        path[: -len(" (deleted)")] if deleted else path
                    )
                    kind = _kind_for_path(normalized_path, db_path)
                    if kind is None:
                        continue
                    key = (pid, kind, normalized_path)
                    if key in seen:
                        continue
                    seen.add(key)
                    files.append(
                        OpenStateDbFile(
                            pid=pid,
                            command=command,
                            fd=str(getattr(opened, "fd", "") or ""),
                            path=normalized_path,
                            kind=kind,
                            inode=(
                                current_inodes.get(kind)
                                if not deleted
                                else None
                            ),
                            link_count=0 if deleted else 1,
                            deleted=deleted,
                        )
                    )
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
            except (psutil.AccessDenied, OSError) as exc:
                # System-owned processes commonly reject handle inspection.
                # Hermes processes run as the current user and remain visible;
                # retain the diagnostic without treating unrelated protected
                # processes as database owners.
                errors.append(str(exc))
    except Exception as exc:
        return StateDbOwnershipReport(
            db_path=db_path,
            available=False,
            files=(),
            current_inodes=current_inodes,
            errors=(f"Windows process ownership probe failed: {exc}",),
        )

    available = not errors
    if not windows and files:
        errors.append(
            "lsof unavailable; psutil found live state.db owner(s) but cannot "
            "verify Unix inode generations"
        )
        available = False

    return StateDbOwnershipReport(
        db_path=db_path,
        available=available,
        files=tuple(files),
        current_inodes=current_inodes,
        errors=tuple(errors),
    )


def inspect_state_db_ownership(
    db_path: Path,
    *,
    runner: Callable[..., Any] | None = None,
    process_iter: Callable[..., Iterable[Any]] | None = None,
) -> StateDbOwnershipReport:
    """Inspect linked and unlinked live owners of one ``state.db``.

    ``lsof`` provides inode/link-count fidelity. When it is absent on Unix,
    ``psutil.open_files`` may prove that a stopped DB has zero owners, but any
    live owner remains unavailable because psutil cannot distinguish an
    unlinked generation after pathname reuse. Windows uses process handles
    because a live SQLite sidecar cannot normally be unlinked and replaced
    behind an existing owner there.
    """

    db_path = db_path.expanduser().resolve(strict=False)
    current_inodes = _current_db_inodes(db_path)
    if sys.platform == "win32":
        return _inspect_psutil_state_db_ownership(
            db_path,
            current_inodes,
            process_iter=process_iter,
            windows=True,
        )

    lsof = shutil.which("lsof")
    if not lsof:
        return _inspect_psutil_state_db_ownership(
            db_path,
            current_inodes,
            process_iter=process_iter,
            windows=False,
        )

    run = runner or subprocess.run
    # Passing absent sidecars to macOS lsof emits a status error even when the
    # main DB probe is otherwise valid. Snapshot only paths that existed when
    # inode metadata was collected. A removal race still fails closed because
    # lsof will then report the vanished path as an actual error.
    linked_paths: list[str] = []
    if "main" in current_inodes:
        linked_paths.append(str(db_path))
    if "wal" in current_inodes:
        linked_paths.append(f"{db_path}-wal")
    if "shm" in current_inodes:
        linked_paths.append(f"{db_path}-shm")
    if "journal" in current_inodes:
        linked_paths.append(f"{db_path}-journal")
    parsed: list[_ParsedLsofFile] = []
    errors: list[str] = []
    ran_any = False
    probe_failed = False

    def _run_lsof(command: list[str]) -> list[_ParsedLsofFile]:
        nonlocal probe_failed, ran_any
        try:
            result = run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=20,
            )
            ran_any = True
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
            errors.append(str(exc))
            probe_failed = True
            return []
        # lsof returns 1 when no files match; stdout remains authoritative.
        stdout = getattr(result, "stdout", "") or ""
        stderr = (getattr(result, "stderr", "") or "").strip()
        returncode = int(getattr(result, "returncode", 0) or 0)
        if returncode not in (0, 1) or stderr:
            detail = stderr or f"lsof exited with status {returncode}"
            errors.append(detail)
            probe_failed = True
        return parse_lsof_field_output(stdout)

    if linked_paths:
        linked = _run_lsof([lsof, "-nP", "-Fpcfikn", "--", *linked_paths])
        parsed.extend(linked)
    # Always inspect unlinked files. A whole state.db/WAL/SHM triplet can be
    # replaced while an old process retains only deleted descriptors, leaving
    # no owner in the linked-path query. Filtering happens below, after the
    # machine-wide scan exposes that otherwise-invisible generation.
    if not probe_failed:
        parsed.extend(_run_lsof([lsof, "-nP", "+L1", "-Fpcfikn"]))

    filtered: list[OpenStateDbFile] = []
    seen: set[tuple[int, str, int | None, str, bool]] = set()
    command_cache: dict[int, str] = {}
    for item in parsed:
        kind = _kind_for_path(item.path, db_path)
        if kind is None:
            continue
        # lsof may report the same SQLite mapping once as a numeric FD and
        # again as ``txt``/memory mapping. One process/inode generation is one
        # diagnostic fact; collapse those aliases to keep Doctor actionable.
        key = (item.pid, kind, item.inode, item.path, item.deleted)
        if key in seen:
            continue
        seen.add(key)
        full_command = command_cache.get(item.pid)
        if full_command is None:
            full_command = _full_process_command(item.pid, item.command)
            command_cache[item.pid] = full_command
        filtered.append(
            OpenStateDbFile(
                pid=item.pid,
                command=full_command,
                fd=item.fd,
                path=item.path,
                kind=kind,
                inode=item.inode,
                link_count=item.link_count,
                deleted=item.deleted,
            )
        )

    return StateDbOwnershipReport(
        db_path=db_path,
        available=ran_any and not probe_failed,
        files=tuple(filtered),
        current_inodes=current_inodes,
        errors=tuple(errors),
    )


def prove_state_db_quiescent(db_path: Path) -> StateDbQuiescence:
    """Prove that no process owns any generation of a SQLite DB cohort.

    This is the mutation boundary used by restore/repair/quarantine paths.
    A missing ownership tool, incomplete scan, linked owner, or deleted old
    generation all fail closed. Callers must re-run this immediately before
    the first rename/replace because an earlier proof can become stale.
    """

    path = Path(db_path).expanduser().resolve(strict=False)
    try:
        report = inspect_state_db_ownership(path)
    except Exception as exc:
        return StateDbQuiescence(
            db_path=path,
            quiescent=False,
            reason=f"ownership probe failed: {exc}",
            report=None,
        )
    if not report.available:
        detail = "; ".join(report.errors) or "ownership probe unavailable"
        return StateDbQuiescence(
            db_path=path,
            quiescent=False,
            reason=detail,
            report=report,
        )
    if report.pids:
        return StateDbQuiescence(
            db_path=path,
            quiescent=False,
            reason=(
                "live SQLite owner(s): "
                + ", ".join(str(pid) for pid in report.pids)
            ),
            report=report,
        )
    if path.name == "state.db":
        home = path.parent
        live_services = _live_service_identity_pids(home)
        if live_services:
            return StateDbQuiescence(
                db_path=path,
                quiescent=False,
                reason=(
                    "live gateway/dashboard/serve identity PID(s): "
                    + ", ".join(str(pid) for pid in live_services)
                ),
                report=report,
            )
        loaded_labels = _loaded_managed_service_labels(home)
        if loaded_labels:
            return StateDbQuiescence(
                db_path=path,
                quiescent=False,
                reason=(
                    "managed launchd service(s) remain loaded and may reopen "
                    "state.db: " + ", ".join(loaded_labels)
                ),
                report=report,
            )
    if report.unlinked_files or report.foreign_inodes or report.split_kinds:
        return StateDbQuiescence(
            db_path=path,
            quiescent=False,
            reason="stale or split SQLite file generation is still open",
            report=report,
        )
    return StateDbQuiescence(
        db_path=path,
        quiescent=True,
        reason="no live SQLite owners",
        report=report,
    )


def _runtime_identity_dir(home: Path) -> Path:
    return home / _IDENTITY_DIR


def _current_process_start_time(pid: int) -> int | None:
    try:
        from gateway.status import _get_process_start_time

        return _get_process_start_time(pid)
    except Exception:
        return None


def _current_source_status() -> dict[str, str | bool | None]:
    try:
        from gateway.code_skew import source_revision_status

        return source_revision_status()
    except Exception:
        return {
            "boot_revision": None,
            "disk_revision": None,
            "code_skew": None,
        }


def write_runtime_identity(
    kind: str,
    *,
    home: Path,
    details: dict[str, Any] | None = None,
) -> Path | None:
    """Persist this process's boot identity for later Doctor correlation."""

    safe_kind = "".join(
        char for char in str(kind).strip().lower() if char.isalnum() or char in "-_"
    )
    if not safe_kind:
        return None
    pid = os.getpid()
    path = _runtime_identity_dir(home) / f"{safe_kind}-{pid}.json"
    payload: dict[str, Any] = {
        "kind": safe_kind,
        "pid": pid,
        "start_time": _current_process_start_time(pid),
        "argv": list(sys.argv),
        "hermes_home": str(home.expanduser().resolve(strict=False)),
        "source": _current_source_status(),
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    if details:
        payload["details"] = dict(details)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_write(path, payload, indent=None, separators=(",", ":"))
        return path
    except Exception:
        return None


def remove_runtime_identity(path: Path | None) -> None:
    """Remove an identity file only when it still belongs to this process."""

    if path is None:
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or int(payload.get("pid", -1)) != os.getpid():
            return
        path.unlink(missing_ok=True)
    except Exception:
        return


def _load_json_dict(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _identity_matches_pid(record: dict[str, Any], pid: int) -> bool:
    try:
        if int(record.get("pid")) != int(pid):
            return False
    except (TypeError, ValueError):
        return False
    recorded_start = record.get("start_time")
    current_start = _current_process_start_time(pid)
    if recorded_start is not None:
        if current_start is None or recorded_start != current_start:
            return False
    return True


def _source_from_record(record: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    source = record.get("source")
    return source if isinstance(source, dict) else {}


def _gateway_runtime_record(home: Path, pid: int) -> dict[str, Any] | None:
    for candidate_home in _identity_search_homes(home):
        record = _load_json_dict(candidate_home / "gateway_state.json")
        if record and _identity_matches_pid(record, pid):
            return record
    return None


def _identity_search_homes(home: Path) -> tuple[Path, ...]:
    """Return profile-local then machine-root homes for identity lookup."""
    resolved = home.expanduser().resolve(strict=False)
    if resolved.parent.name == "profiles":
        return (resolved, resolved.parent.parent)
    return (resolved,)


def _live_service_identity_pids(home: Path) -> tuple[int, ...]:
    """Return live gateway/dashboard/serve identities for one profile."""

    pids: set[int] = set()
    for candidate_home in _identity_search_homes(home):
        gateway = _load_json_dict(candidate_home / "gateway_state.json")
        if isinstance(gateway, dict):
            try:
                pid = int(gateway.get("pid"))
            except (TypeError, ValueError):
                pid = -1
            if pid > 0 and _identity_matches_pid(gateway, pid):
                pids.add(pid)
        directory = _runtime_identity_dir(candidate_home)
        try:
            paths = tuple(directory.glob("*.json"))
        except OSError:
            paths = ()
        for path in paths:
            record = _load_json_dict(path)
            if not isinstance(record, dict):
                continue
            if record.get("kind") not in {"gateway", "dashboard", "serve"}:
                continue
            try:
                pid = int(record.get("pid"))
            except (TypeError, ValueError):
                continue
            if pid > 0 and _identity_matches_pid(record, pid):
                pids.add(pid)
    return tuple(sorted(pids))


def _loaded_managed_service_labels(home: Path) -> tuple[str, ...]:
    """Return relevant loaded launchd services that could reopen state.db."""

    if sys.platform != "darwin":
        return ()
    try:
        import plistlib
        import pwd

        user_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    except Exception:
        return ("launchd-service-state-unavailable",)

    resolved_home = str(home.expanduser().resolve(strict=False))
    # The managed default launchd services belong to the account's physical
    # ~/.hermes root, not an HERMES_HOME override used by tests/import staging.
    default_home = str((user_home / ".hermes").resolve(strict=False))
    launch_agents = user_home / "Library" / "LaunchAgents"
    labels: set[str] = set()
    try:
        plists = tuple(launch_agents.glob("*.plist"))
    except OSError:
        return ("launchd-service-state-unavailable",)

    for plist_path in plists:
        try:
            with plist_path.open("rb") as handle:
                payload = plistlib.load(handle)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        label = str(payload.get("Label") or "")
        lowered = label.lower()
        if not label or not (
            lowered.startswith("ai.hermes.gateway")
            or lowered.startswith("ai.hermes.dashboard")
            or lowered.startswith("ai.hermes.serve")
        ):
            continue
        environment = payload.get("EnvironmentVariables")
        env_home = (
            str(environment.get("HERMES_HOME") or "")
            if isinstance(environment, dict)
            else ""
        )
        arguments = payload.get("ProgramArguments")
        arg_text = (
            "\n".join(str(item) for item in arguments)
            if isinstance(arguments, list)
            else str(arguments or "")
        )
        exact_env_token = f"HERMES_HOME={resolved_home}"
        relevant = env_home == resolved_home or exact_env_token in arg_text
        has_any_home_override = bool(env_home) or "HERMES_HOME=" in arg_text
        if (
            not relevant
            and resolved_home == default_home
            and not has_any_home_override
            and (
                label == "ai.hermes.gateway"
                or lowered.startswith("ai.hermes.dashboard")
                or lowered.startswith("ai.hermes.serve")
            )
        ):
            relevant = True
        if not relevant:
            continue
        loaded = False
        for domain in (f"gui/{os.getuid()}", f"user/{os.getuid()}"):
            try:
                result = subprocess.run(
                    ["launchctl", "print", f"{domain}/{label}"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=10,
                )
            except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
                return ("launchd-service-state-unavailable",)
            if result.returncode == 0:
                loaded = True
                break
        if loaded:
            labels.add(label)
    return tuple(sorted(labels))


def _process_identity_record(home: Path, pid: int) -> dict[str, Any] | None:
    for candidate_home in _identity_search_homes(home):
        directory = _runtime_identity_dir(candidate_home)
        try:
            candidates: Iterable[Path] = directory.glob(f"*-{int(pid)}.json")
            for path in candidates:
                record = _load_json_dict(path)
                if record and _identity_matches_pid(record, pid):
                    return record
        except (OSError, ValueError):
            continue
    return None


def _owner_role(command: str, identity: dict[str, Any] | None) -> str:
    if identity and identity.get("kind") in {"dashboard", "serve"}:
        return str(identity["kind"])
    lowered = (command or "").lower()
    if " dashboard" in lowered or "hermes_cli.main dashboard" in lowered:
        return "dashboard"
    if " serve" in lowered or "hermes_cli.main serve" in lowered:
        return "serve"
    if " gateway" in lowered or "gateway.run" in lowered:
        return "gateway"
    return "other"


def summarize_state_db_owners(
    report: StateDbOwnershipReport,
    *,
    home: Path,
) -> tuple[StateDbOwner, ...]:
    """Correlate live file owners with persisted boot revisions."""

    current_source = _current_source_status()
    current_disk_revision = current_source.get("disk_revision")
    by_pid: dict[int, str] = {}
    for item in report.files:
        by_pid.setdefault(item.pid, item.command)

    owners: list[StateDbOwner] = []
    for pid in sorted(by_pid):
        identity = _process_identity_record(home, pid)
        source = _source_from_record(identity)
        if not source:
            source = _source_from_record(_gateway_runtime_record(home, pid))
        boot_revision = source.get("boot_revision")
        owners.append(
            StateDbOwner(
                pid=pid,
                command=by_pid[pid],
                role=_owner_role(by_pid[pid], identity),
                boot_revision=(
                    str(boot_revision) if boot_revision not in (None, "") else None
                ),
                disk_revision=(
                    str(current_disk_revision)
                    if current_disk_revision not in (None, "")
                    else None
                ),
            )
        )
    return tuple(owners)
