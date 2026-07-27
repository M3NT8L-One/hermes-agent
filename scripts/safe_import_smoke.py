#!/usr/bin/env python3
"""Run Hermes import validation with SQLite and live-state safeguards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from hermes_cli.sqlite_runtime import run_isolated_import_smoke


DEFAULT_MODULES = (
    "hermes_state",
    "tools.process_registry",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Probe an exact Python interpreter, reject WAL-reset-vulnerable "
            "SQLite, and import modules under a temporary HOME/HERMES_HOME."
        )
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Exact Python executable to validate (default: this interpreter).",
    )
    parser.add_argument(
        "--cwd",
        type=Path,
        default=REPO_ROOT,
        help="Working directory for the isolated import child.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit one machine-readable result object.",
    )
    parser.add_argument(
        "modules",
        nargs="*",
        default=list(DEFAULT_MODULES),
        help="Modules to import (default: hermes_state tools.process_registry).",
    )
    args = parser.parse_args(argv)

    healthy, detail, info = run_isolated_import_smoke(
        args.python,
        args.modules,
        cwd=args.cwd,
    )
    payload = {
        "healthy": healthy,
        "detail": detail,
        "python": str(info.executable) if info is not None else str(args.python),
        "sqlite": info.sqlite_version_string if info is not None else "",
        "wal_reset_vulnerable": (
            info.wal_reset_vulnerable if info is not None else None
        ),
        "modules": list(args.modules),
        "live_hermes_home_exposed": False,
    }
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    elif healthy:
        print(
            "SAFE: isolated imports passed with "
            f"SQLite {payload['sqlite']} ({', '.join(args.modules)})"
        )
    else:
        print(f"REFUSED: {detail}", file=sys.stderr)
    return 0 if healthy else 2


if __name__ == "__main__":
    raise SystemExit(main())
