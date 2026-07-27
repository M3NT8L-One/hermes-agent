"""Import-safe helpers for inspecting a Python interpreter's linked SQLite.

This module intentionally depends only on the standard library.  Installer and
update code must be able to use it before Hermes' third-party dependencies are
healthy.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def _version_tuple(parts: Iterable[object]) -> tuple[int, int, int]:
    values = [int(part) for part in parts]
    values.extend([0] * (3 - len(values)))
    return tuple(values[:3])


def is_sqlite_wal_reset_vulnerable(
    version_info: tuple[int, ...],
) -> bool:
    """Return whether *version_info* contains SQLite's WAL-reset bug."""
    info = _version_tuple(version_info)
    if info < (3, 7, 0):
        return False
    if info >= (3, 51, 3):
        return False
    if (3, 50, 7) <= info < (3, 51, 0):
        return False
    if (3, 44, 6) <= info < (3, 45, 0):
        return False
    return True


@dataclass(frozen=True)
class SQLiteRuntimeInfo:
    """SQLite details reported by one exact Python executable."""

    executable: Path
    base_prefix: Path
    python_version: tuple[int, int, int]
    sqlite_version: tuple[int, int, int]
    sqlite_version_string: str
    sqlite_source_id: str

    @property
    def wal_reset_vulnerable(self) -> bool:
        return is_sqlite_wal_reset_vulnerable(self.sqlite_version)


_PROBE_SCRIPT = """
import json
import sqlite3
import sys

conn = sqlite3.connect(":memory:")
try:
    row = conn.execute("SELECT sqlite_source_id()").fetchone()
finally:
    conn.close()

print(json.dumps({
    "base_prefix": sys.base_prefix,
    "executable": sys.executable,
    "python_version": list(sys.version_info[:3]),
    "sqlite_version": list(sqlite3.sqlite_version_info),
    "sqlite_version_string": sqlite3.sqlite_version,
    "sqlite_source_id": str(row[0]) if row and row[0] is not None else "",
}))
"""

_IMPORT_SMOKE_SCRIPT = """
import importlib
import sys

for module_name in sys.argv[1:]:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        print(
            f"{module_name}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
"""

_PYTHON_ENV_OVERRIDES = (
    "CONDA_DEFAULT_ENV",
    "CONDA_PREFIX",
    "PYTHONHOME",
    "PYTHONPATH",
    "UV_PROJECT_ENVIRONMENT",
    "UV_PYTHON",
    "VIRTUAL_ENV",
)


def _sanitized_python_env() -> dict[str, str]:
    """Return an environment free of interpreter-selection overrides."""
    env = dict(os.environ)
    for key in _PYTHON_ENV_OVERRIDES:
        env.pop(key, None)
    return env


def probe_sqlite_runtime(
    python: str | Path,
    *,
    timeout: float = 30.0,
) -> SQLiteRuntimeInfo | None:
    """Probe SQLite in *python*, never the caller's linked SQLite.

    ``None`` means the interpreter could not be executed or returned malformed
    data.  The child runs isolated from inherited Python path overrides.
    """
    executable = Path(python)
    env = _sanitized_python_env()
    try:
        result = subprocess.run(
            [str(executable), "-I", "-c", _PROBE_SCRIPT],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout)
        return SQLiteRuntimeInfo(
            executable=Path(str(payload["executable"])),
            base_prefix=Path(str(payload["base_prefix"])),
            python_version=_version_tuple(payload["python_version"]),
            sqlite_version=_version_tuple(payload["sqlite_version"]),
            sqlite_version_string=str(payload["sqlite_version_string"]),
            sqlite_source_id=str(payload.get("sqlite_source_id", "")),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def run_isolated_import_smoke(
    python: str | Path,
    modules: Iterable[str],
    *,
    cwd: str | Path | None = None,
    timeout: float = 90.0,
) -> tuple[bool, str, SQLiteRuntimeInfo | None]:
    """Import *modules* without exposing live Hermes state to the child.

    The exact target interpreter is probed first and rejected when its linked
    SQLite contains the WAL-reset corruption bug.  A passing interpreter then
    runs with ``HERMES_HOME``, ``HOME``, and ``USERPROFILE`` redirected into
    one temporary directory, so both profile-aware and legacy home-relative
    imports are unable to open the operator's live ``state.db``.

    Returns ``(healthy, detail, runtime_info)`` and never raises for ordinary
    execution or import failures.
    """
    # Absolutize once before either child runs, but deliberately do not resolve
    # symlinks: venv/bin/python commonly points at a base interpreter, and
    # executing the resolved target would lose the venv's site-packages.
    executable = Path(os.path.abspath(os.path.expanduser(os.fspath(python))))
    requested = tuple(str(module).strip() for module in modules if str(module).strip())
    if not requested:
        return False, "no import modules requested", None

    info = probe_sqlite_runtime(executable, timeout=min(timeout, 30.0))
    if info is None:
        return False, f"could not execute {executable}", None
    if info.wal_reset_vulnerable:
        return (
            False,
            f"interpreter links vulnerable SQLite {info.sqlite_version_string}",
            info,
        )

    env = _sanitized_python_env()
    try:
        with tempfile.TemporaryDirectory(prefix="hermes-import-smoke-") as root:
            isolated_home = Path(root) / "home"
            isolated_hermes = isolated_home / ".hermes"
            isolated_hermes.mkdir(parents=True)
            env.update(
                {
                    "HOME": str(isolated_home),
                    "USERPROFILE": str(isolated_home),
                    "HERMES_HOME": str(isolated_hermes),
                    "HERMES_IMPORT_SMOKE": "1",
                }
            )
            result = subprocess.run(
                [
                    str(executable),
                    "-I",
                    "-c",
                    _IMPORT_SMOKE_SCRIPT,
                    *requested,
                ],
                cwd=str(cwd) if cwd is not None else None,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc), info

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "import smoke failed").strip()
        last_line = detail.splitlines()[-1] if detail else "import smoke failed"
        return False, last_line, info
    return True, "", info
