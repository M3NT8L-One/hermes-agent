"""Behavioral tests for exact-interpreter SQLite runtime inspection."""

from __future__ import annotations

import json
import os
import shlex
import sqlite3
import sys
from pathlib import Path

import pytest

from hermes_cli.sqlite_runtime import (
    is_sqlite_wal_reset_vulnerable,
    probe_sqlite_runtime,
    run_isolated_import_smoke,
)


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ((3, 6, 23), False),
        ((3, 7, 0), True),
        ((3, 44, 5), True),
        ((3, 44, 6), False),
        ((3, 45, 0), True),
        ((3, 50, 6), True),
        ((3, 50, 7), False),
        ((3, 51, 2), True),
        ((3, 51, 3), False),
        ((3, 53, 1), False),
    ],
)
def test_wal_reset_vulnerability_matrix(
    version: tuple[int, ...],
    expected: bool,
) -> None:
    assert is_sqlite_wal_reset_vulnerable(version) is expected


def test_probe_reports_the_requested_interpreters_linked_sqlite() -> None:
    info = probe_sqlite_runtime(sys.executable)

    assert info is not None
    assert info.executable.resolve() == Path(sys.executable).resolve()
    assert info.base_prefix.resolve() == Path(sys.base_prefix).resolve()
    assert info.python_version == sys.version_info[:3]
    assert info.sqlite_version == sqlite3.sqlite_version_info
    assert info.sqlite_version_string == sqlite3.sqlite_version

    with sqlite3.connect(":memory:") as conn:
        source_id = conn.execute("SELECT sqlite_source_id()").fetchone()[0]
    assert info.sqlite_source_id == source_id


@pytest.mark.skipif(os.name == "nt", reason="uses a POSIX executable probe stub")
def test_probe_uses_child_payload_and_sanitizes_python_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_python = tmp_path / "reported-python"
    payload = {
        "base_prefix": str(tmp_path / "reported-base"),
        "executable": str(fake_python),
        "python_version": [3, 11, 15],
        "sqlite_version": [9, 8, 7],
        "sqlite_version_string": "9.8.7-child",
        "sqlite_source_id": "child-source-id",
    }
    fake_python.write_text(
        "\n".join([
            "#!/bin/sh",
            '[ "$1" = "-I" ] && [ "$2" = "-c" ] || exit 10',
            '[ -z "${PYTHONHOME+x}" ] || exit 11',
            '[ -z "${PYTHONPATH+x}" ] || exit 12',
            f"printf '%s\\n' {shlex.quote(json.dumps(payload))}",
        ])
        + "\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    monkeypatch.setenv("PYTHONHOME", str(tmp_path / "poison-home"))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "poison-path"))

    info = probe_sqlite_runtime(fake_python)

    assert info is not None
    assert info.executable == fake_python
    assert info.base_prefix == tmp_path / "reported-base"
    assert info.sqlite_version == (9, 8, 7)
    assert info.sqlite_version_string == "9.8.7-child"
    assert info.sqlite_source_id == "child-source-id"


@pytest.mark.skipif(os.name == "nt", reason="uses a POSIX executable probe stub")
def test_import_smoke_redirects_all_home_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_python = tmp_path / "reported-python"
    record = tmp_path / "smoke-environment.json"
    live_home = tmp_path / "operator-home"
    live_hermes = live_home / ".hermes"
    live_hermes.mkdir(parents=True)
    payload = {
        "base_prefix": str(tmp_path / "reported-base"),
        "executable": str(fake_python),
        "python_version": [3, 11, 15],
        "sqlite_version": [3, 53, 1],
        "sqlite_version_string": "3.53.1",
        "sqlite_source_id": "safe-child-source",
    }
    fake_python.write_text(
        "\n".join(
            [
                "#!/bin/sh",
                'if [ "$#" -eq 3 ]; then',
                f"  printf '%s\\n' {shlex.quote(json.dumps(payload))}",
                "  exit 0",
                "fi",
                '[ "$1" = "-I" ] && [ "$2" = "-c" ] || exit 20',
                '[ "$4" = "tools.process_registry" ] || exit 21',
                '[ "$HERMES_IMPORT_SMOKE" = "1" ] || exit 22',
                '[ "$HERMES_HOME" != "$LIVE_HERMES_HOME" ] || exit 23',
                '[ "$HOME" != "$LIVE_HOME" ] || exit 24',
                '[ "$USERPROFILE" = "$HOME" ] || exit 25',
                'case "$HERMES_HOME" in "$HOME"/.hermes) ;; *) exit 26 ;; esac',
                (
                    "printf '{\"home\":\"%s\",\"hermes_home\":\"%s\"}\\n' "
                    '"$HOME" "$HERMES_HOME" > "$SMOKE_RECORD"'
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    monkeypatch.setenv("HOME", str(live_home))
    monkeypatch.setenv("USERPROFILE", str(live_home))
    monkeypatch.setenv("HERMES_HOME", str(live_hermes))
    monkeypatch.setenv("LIVE_HOME", str(live_home))
    monkeypatch.setenv("LIVE_HERMES_HOME", str(live_hermes))
    monkeypatch.setenv("SMOKE_RECORD", str(record))

    healthy, detail, info = run_isolated_import_smoke(
        fake_python,
        ["tools.process_registry"],
        cwd=tmp_path,
    )

    assert healthy is True
    assert detail == ""
    assert info is not None
    assert info.sqlite_version == (3, 53, 1)
    isolated = json.loads(record.read_text(encoding="utf-8"))
    assert isolated["home"] != str(live_home)
    assert isolated["hermes_home"] != str(live_hermes)
    assert isolated["hermes_home"] == str(Path(isolated["home"]) / ".hermes")
    assert not (live_hermes / "state.db").exists()


@pytest.mark.skipif(os.name == "nt", reason="uses POSIX executable symlinks")
def test_import_smoke_absolutizes_relative_venv_python_before_child_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_python = tmp_path / "runtime" / "python"
    real_python.parent.mkdir()
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    record = tmp_path / "executed-as.txt"
    child_cwd = tmp_path / "different-cwd"
    child_cwd.mkdir()
    payload = {
        "base_prefix": str(tmp_path / "reported-base"),
        "executable": str(venv_python),
        "python_version": [3, 11, 15],
        "sqlite_version": [3, 53, 1],
        "sqlite_version_string": "3.53.1",
        "sqlite_source_id": "safe-child-source",
    }
    real_python.write_text(
        "\n".join(
            [
                "#!/bin/sh",
                'if [ "$#" -eq 3 ]; then',
                f"  printf '%s\\n' {shlex.quote(json.dumps(payload))}",
                "  exit 0",
                "fi",
                '[ "$1" = "-I" ] && [ "$2" = "-c" ] || exit 20',
                '[ "$4" = "demo.module" ] || exit 21',
                'printf "%s\\n" "$0" > "$EXECUTED_AS_RECORD"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    real_python.chmod(0o755)
    venv_python.symlink_to(real_python)
    monkeypatch.setenv("EXECUTED_AS_RECORD", str(record))
    monkeypatch.chdir(tmp_path)

    healthy, detail, info = run_isolated_import_smoke(
        Path("venv/bin/python"),
        ["demo.module"],
        cwd=child_cwd,
    )

    assert healthy is True
    assert detail == ""
    assert info is not None
    # The child ran through the absolute venv entry point, not a path relative
    # to child_cwd and not the symlink-resolved base interpreter.
    assert record.read_text(encoding="utf-8").strip() == str(venv_python)


def test_import_smoke_refuses_vulnerable_runtime_before_import(
    tmp_path: Path,
) -> None:
    from hermes_cli.sqlite_runtime import SQLiteRuntimeInfo

    python = tmp_path / "python"
    vulnerable = SQLiteRuntimeInfo(
        executable=python,
        base_prefix=tmp_path,
        python_version=(3, 11, 14),
        sqlite_version=(3, 50, 4),
        sqlite_version_string="3.50.4",
        sqlite_source_id="vulnerable-source",
    )
    from unittest.mock import patch

    with patch(
        "hermes_cli.sqlite_runtime.probe_sqlite_runtime",
        return_value=vulnerable,
    ), patch("hermes_cli.sqlite_runtime.subprocess.run") as smoke_run:
        healthy, detail, info = run_isolated_import_smoke(python, ["hermes_state"])

    assert healthy is False
    assert detail == "interpreter links vulnerable SQLite 3.50.4"
    assert info == vulnerable
    smoke_run.assert_not_called()
