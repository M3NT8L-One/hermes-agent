"""Operator-configured capability grants for delegated child missions.

Capability grants are deliberately narrow.  They authorize only exact local
service reload grammars and content-bound project controller commands.  The
model selects an opaque name; trusted config owns every path, service, command,
and budget.  Hardline blocks and user deny rules remain authoritative.
"""

from __future__ import annotations

import copy
import fnmatch
import hashlib
import json
import os
import re
import shlex
import shutil
from pathlib import Path
from typing import Any, Optional


SUPPORTED_OPERATIONS = frozenset({"controller", "service_reload"})

CONTROLLER_CONTRACT_BOUNDARIES = (
    "credential_or_secret_change",
    "external_publish_or_message",
    "git_history_destruction_or_force_push",
    "live_money_or_account_mutation",
    "host_or_device_destruction",
)

_ALLOWED_GRANT_KEYS = frozenset(
    {
        "allowed_roots",
        "allowed_services",
        "allowed_operations",
        "service_commands",
        "controller_commands",
        "max_iterations",
        "child_timeout_seconds",
    }
)
_SAFE_SERVICE_RE = re.compile(r"^[A-Za-z0-9_.:@/+-]+$")
_SHELL_META_RE = re.compile(r"[$`;&|<>{}()\[\]*?\\\n\r]")
_SERVICE_PROGRAMS = frozenset({"launchctl", "systemctl"})
_SERVICE_LIFECYCLE_HINT_RE = re.compile(
    r"(?:^|[\s;&|()'\"`])(?:[^\s;&|()'\"`]+/)*"
    r"(?:"
    r"launchctl\s+(?:kickstart|start|stop|bootout|bootstrap|load|unload)\b|"
    r"systemctl\b[^\n;&|]{0,160}\b(?:restart|start|stop|reload|enable|disable)\b|"
    r"docker\s+(?:(?:compose\s+)?(?:restart|start|stop|kill|rm|up|down))\b|"
    r"hermes\b[^\n;&|]{0,80}\bgateway\b[^\n;&|]{0,80}\b(?:restart|start|stop)\b"
    r")",
    re.IGNORECASE,
)
_SERVICE_MANAGER_HINT_RE = re.compile(
    r"(?:launchct(?:l|\[[^\]\s]+\])|systemct(?:l|\[[^\]\s]+\])|"
    r"docke(?:r|\[[^\]\s]+\])|herme(?:s|\[[^\]\s]+\]))",
    re.IGNORECASE,
)
_SERVICE_MUTATION_HINT_RE = re.compile(
    r"\b(?:kickstart|restart|start|stop|reload|enable|disable|bootout|"
    r"bootstrap|load|unload|kill|rm|up|down)\b",
    re.IGNORECASE,
)
_CONTROLLER_INTERPRETERS = frozenset(
    {"python", "python2", "python3", "ruby", "perl", "node"}
)
_PYTHON_CONTROLLER_FLAGS = ("-I", "-S")
_INTERACTIVE_SHELLS = frozenset(
    {"bash", "dash", "fish", "ksh", "sh", "zsh"}
)
_INTERACTIVE_REPLS = frozenset(
    {"irb", "node", "perl", "python", "python2", "python3", "ruby"}
)
_PROTECTED_COMPONENTS = frozenset(
    {
        ".ssh",
        ".netrc",
        ".pgpass",
        ".npmrc",
        ".pypirc",
        "auth.json",
        "credential.json",
        "credentials.json",
        "secret.json",
        "secrets.json",
        "token.json",
        "tokens.json",
        "access_token.json",
        "refresh-token.json",
        "refresh_token.json",
        "password.txt",
        "id_rsa",
        "id_dsa",
        "id_ed25519",
        ".git",
    }
)
def _canonical(value: Any) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _positive_int(value: Any, *, field: str) -> int:
    if value in (None, "", 0):
        return 0
    if isinstance(value, bool):
        raise ValueError(f"delegation capability {field} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"delegation capability {field} must be a positive integer"
        ) from exc
    if parsed < 1:
        raise ValueError(f"delegation capability {field} must be a positive integer")
    return parsed


def _string_list(value: Any, *, field: str) -> list[str]:
    if value in (None, []):
        return []
    if not isinstance(value, list):
        raise ValueError(f"delegation capability {field} must be a list")
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"delegation capability {field} values must be strings")
        text = item.strip()
        if not text:
            raise ValueError(f"delegation capability {field} contains an empty value")
        if text not in normalized:
            normalized.append(text)
    return normalized


def _inside_root(path: Path, roots: list[str], *, allow_root: bool = True) -> bool:
    resolved = path.resolve(strict=False)
    for root_value in roots:
        root = Path(root_value).resolve(strict=False)
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        if allow_root or resolved != root:
            return True
    return False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_controller_interpreter(program_name: str) -> bool:
    return program_name in _CONTROLLER_INTERPRETERS or bool(
        re.fullmatch(r"python\d+(?:\.\d+)?", program_name)
    )


def _is_python_interpreter(program_name: str) -> bool:
    if program_name in {"python", "python3"}:
        return True
    match = re.fullmatch(r"python(\d+)(?:\.(\d+))?", program_name)
    if not match:
        return False
    major = int(match.group(1))
    minor = int(match.group(2) or 0)
    return major > 3 or (major == 3 and minor >= 4)


def _normalize_service_commands(value: Any) -> tuple[list[str], list[dict[str, str]]]:
    commands = _string_list(value, field="service_commands")
    identities: list[dict[str, str]] = []
    for command in commands:
        path = Path(command)
        if not path.is_absolute() or path.name.lower() not in _SERVICE_PROGRAMS:
            raise ValueError(
                "delegation capability service_commands must be absolute paths "
                "to launchctl or systemctl"
            )
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise ValueError(
                f"delegation capability service executable does not exist: {path}"
            ) from exc
        if not resolved.is_file():
            raise ValueError(
                "delegation capability service executable must resolve to a file"
            )
        identities.append(
            {
                "command_path": str(path),
                "resolved_path": str(resolved),
                "sha256": _sha256_file(resolved),
                "program": path.name.lower(),
            }
        )
    return commands, identities


def _normalize_controller_commands(
    value: Any,
    *,
    roots: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if value in (None, []):
        return [], []
    if not isinstance(value, list):
        raise ValueError("delegation capability controller_commands must be a list")

    normalized: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    for entry in value:
        if not isinstance(entry, dict):
            raise ValueError(
                "delegation capability controller command must be an argv/options mapping"
            )
        unknown = sorted(
            set(entry)
            - {
                "argv",
                "flag_options",
                "value_options",
                "repeatable_value_options",
                "fixed_value_options",
                "bound_files",
            }
        )
        if unknown:
            raise ValueError(
                "delegation capability controller command has unknown keys: "
                + ", ".join(unknown)
            )
        raw_argv = entry.get("argv")
        if not isinstance(raw_argv, list):
            raise ValueError(
                "delegation capability controller command argv must be a list"
            )
        tokens = [str(token).strip() for token in raw_argv]
        if not tokens or any(not token for token in tokens):
            raise ValueError("delegation capability controller command cannot be empty")
        if any(_SHELL_META_RE.search(token) for token in tokens):
            raise ValueError(
                "delegation capability controller command must be plain argv"
            )

        flag_options = _string_list(
            entry.get("flag_options"), field="controller flag_options"
        )
        value_options = _string_list(
            entry.get("value_options"), field="controller value_options"
        )
        repeatable = _string_list(
            entry.get("repeatable_value_options"),
            field="controller repeatable_value_options",
        )
        raw_fixed = entry.get("fixed_value_options") or {}
        if not isinstance(raw_fixed, dict):
            raise ValueError(
                "delegation capability controller fixed_value_options must be a mapping"
            )
        fixed: dict[str, list[str]] = {}
        for option, choices in raw_fixed.items():
            if not isinstance(option, str):
                raise ValueError("delegation capability controller option names must be strings")
            fixed[option] = _string_list(
                choices, field=f"controller fixed choices for {option}"
            )
            if not fixed[option]:
                raise ValueError(
                    "delegation capability controller fixed choices cannot be empty"
                )
        option_names = set(flag_options) | set(value_options) | set(fixed)
        if any(
            not re.fullmatch(r"--[a-z0-9][a-z0-9-]*", option)
            for option in option_names
        ):
            raise ValueError(
                "delegation capability controller options must be long --kebab-case names"
            )
        if set(flag_options) & (set(value_options) | set(fixed)) or set(
            value_options
        ) & set(fixed):
            raise ValueError(
                "delegation capability controller option classes must not overlap"
            )
        if not set(repeatable).issubset(set(value_options) | set(fixed)):
            raise ValueError(
                "delegation capability repeatable options must also accept values"
            )

        program = Path(tokens[0])
        if not program.is_absolute():
            raise ValueError(
                "delegation capability controller executable must be an absolute path"
            )
        program_name = program.name.lower()
        try:
            resolved_program = program.resolve(strict=True)
        except OSError as exc:
            raise ValueError(
                f"delegation capability controller executable does not exist: {program}"
            ) from exc
        if not resolved_program.is_file():
            raise ValueError(
                "delegation capability controller executable must resolve to a file"
            )
        if _is_controller_interpreter(program_name):
            if not _is_python_interpreter(program_name):
                raise ValueError(
                    "delegation capability interpreted controllers must use Python "
                    "with exact -I -S isolation flags"
                )
            script_index = 1
            while script_index < len(tokens) and tokens[script_index].startswith("-"):
                script_index += 1
            if tuple(tokens[1:script_index]) != _PYTHON_CONTROLLER_FLAGS:
                raise ValueError(
                    "delegation capability Python controllers must use exact -I -S "
                    "isolation flags before the script"
                )
            if len(tokens) < script_index + 2:
                raise ValueError(
                    "delegation capability interpreter controller commands must "
                    "include an absolute script and an allowed controller verb"
                )
            identity_path = Path(tokens[script_index])
        else:
            if len(tokens) < 2:
                raise ValueError(
                    "delegation capability controller commands must include an allowed verb"
                )
            identity_path = program
        if not identity_path.is_absolute():
            raise ValueError(
                "delegation capability controller script must be an absolute path"
            )
        try:
            resolved_identity = identity_path.resolve(strict=True)
        except OSError as exc:
            raise ValueError(
                f"delegation capability controller does not exist: {identity_path}"
            ) from exc
        if not resolved_identity.is_file() or not _inside_root(
            resolved_identity, roots
        ):
            raise ValueError(
                "delegation capability controller must resolve to a file inside allowed_roots"
            )

        bound_files = _string_list(
            entry.get("bound_files"), field="controller bound_files"
        )
        bound_file_identities: list[dict[str, str]] = []
        for bound_file in bound_files:
            bound_path = Path(bound_file)
            if not bound_path.is_absolute():
                raise ValueError(
                    "delegation capability controller bound_files must be absolute paths"
                )
            try:
                resolved_bound = bound_path.resolve(strict=True)
            except OSError as exc:
                raise ValueError(
                    f"delegation capability controller bound file does not exist: {bound_path}"
                ) from exc
            if not resolved_bound.is_file() or not _inside_root(
                resolved_bound, roots
            ):
                raise ValueError(
                    "delegation capability controller bound file must resolve "
                    "inside allowed_roots"
                )
            bound_file_identities.append(
                {
                    "command_path": str(bound_path),
                    "resolved_path": str(resolved_bound),
                    "sha256": _sha256_file(resolved_bound),
                }
            )

        command_spec = {
            "argv": tokens,
            "flag_options": flag_options,
            "value_options": value_options,
            "repeatable_value_options": repeatable,
            "fixed_value_options": fixed,
            "bound_files": bound_files,
        }
        if any(existing.get("argv") == tokens for existing in normalized):
            raise ValueError(
                "delegation capability controller command argv prefixes must be unique"
            )
        if command_spec not in normalized:
            normalized.append(command_spec)
            identities.append(
                {
                    "argv": tokens,
                    "arg_schema": {
                        key: copy.deepcopy(value)
                        for key, value in command_spec.items()
                        if key != "argv"
                    },
                    "executable_command_path": str(program),
                    "executable_path": str(resolved_program),
                    "executable_sha256": _sha256_file(resolved_program),
                    "identity_command_path": str(identity_path),
                    "identity_path": str(resolved_identity),
                    "identity_sha256": _sha256_file(resolved_identity),
                    "bound_file_identities": bound_file_identities,
                }
            )
    return normalized, identities


def configured_capability_grant_names(cfg: Optional[dict] = None) -> list[str]:
    cfg = cfg if isinstance(cfg, dict) else {}
    raw = cfg.get("capability_grants")
    if not isinstance(raw, dict):
        return []
    return sorted(
        str(name)
        for name, grant in raw.items()
        if str(name).strip() and isinstance(grant, dict)
    )


def _normalize_grant(name: str, raw: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(raw) - _ALLOWED_GRANT_KEYS)
    if unknown:
        raise ValueError(
            f"delegation capability_grant {name!r} has unknown keys: "
            + ", ".join(unknown)
        )

    roots = _string_list(raw.get("allowed_roots"), field="allowed_roots")
    normalized_roots: list[str] = []
    for root in roots:
        expanded = Path(os.path.expandvars(os.path.expanduser(root)))
        if not expanded.is_absolute():
            raise ValueError(
                f"delegation capability_grant {name!r} allowed_roots must be absolute"
            )
        resolved = str(expanded.resolve(strict=False))
        if resolved not in normalized_roots:
            normalized_roots.append(resolved)

    services = _string_list(raw.get("allowed_services"), field="allowed_services")
    if any(not _SAFE_SERVICE_RE.fullmatch(service) for service in services):
        raise ValueError(
            f"delegation capability_grant {name!r} has an invalid service label"
        )
    operations = _string_list(
        raw.get("allowed_operations"), field="allowed_operations"
    )
    unsupported = sorted(set(operations) - SUPPORTED_OPERATIONS)
    if unsupported:
        raise ValueError(
            f"delegation capability_grant {name!r} has unsupported operations: "
            + ", ".join(unsupported)
        )

    service_commands, service_identities = _normalize_service_commands(
        raw.get("service_commands")
    )
    controller_commands, controller_identities = _normalize_controller_commands(
        raw.get("controller_commands"), roots=normalized_roots
    )
    if "controller" in operations and not controller_commands:
        raise ValueError(
            f"delegation capability_grant {name!r} enables controller but "
            "controller_commands is empty"
        )
    if controller_commands and "controller" not in operations:
        raise ValueError(
            f"delegation capability_grant {name!r} configures controllers without "
            "the controller operation"
        )
    if "service_reload" in operations and (
        not normalized_roots or not services or not service_commands
    ):
        raise ValueError(
            f"delegation capability_grant {name!r} enables service_reload but "
            "allowed_roots, allowed_services, or service_commands is empty"
        )
    if service_commands and "service_reload" not in operations:
        raise ValueError(
            f"delegation capability_grant {name!r} configures service_commands "
            "without the service_reload operation"
        )

    normalized = {
        "name": name,
        "allowed_roots": normalized_roots,
        "allowed_services": services,
        "allowed_operations": operations,
        "service_commands": service_commands,
        "service_identities": service_identities,
        "controller_commands": controller_commands,
        "controller_identities": controller_identities,
        "max_iterations": _positive_int(
            raw.get("max_iterations"), field="max_iterations"
        ),
        "child_timeout_seconds": _positive_int(
            raw.get("child_timeout_seconds"), field="child_timeout_seconds"
        ),
        "controller_contract_boundaries": list(CONTROLLER_CONTRACT_BOUNDARIES),
    }
    digest_payload = {key: value for key, value in normalized.items() if key != "name"}
    normalized["grant_digest"] = "sha256:" + hashlib.sha256(
        _canonical(digest_payload).encode("utf-8")
    ).hexdigest()
    return normalized


def select_capability_grant(
    cfg: dict,
    requested_name: Optional[str] = None,
    *,
    inherited_name: Optional[str] = None,
    inherited_grant: Optional[dict[str, Any]] = None,
    inherit_from_parent: bool = False,
) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    """Resolve an opaque grant name without allowing nested escalation."""
    cfg = cfg if isinstance(cfg, dict) else {}
    requested = str(requested_name).strip() if requested_name is not None else ""
    inherited = str(inherited_name or "").strip()
    if inherit_from_parent and not inherited:
        if requested:
            raise ValueError(
                "Nested delegation cannot add a capability_grant when the "
                "parent mission has none."
            )
        return None, None
    if inherited:
        if requested and requested != inherited:
            raise ValueError(
                f"Nested delegation cannot expand capability_grant from "
                f"{inherited!r} to {requested!r}; child missions inherit the "
                "parent grant."
            )
        if isinstance(inherited_grant, dict):
            return inherited, copy.deepcopy(inherited_grant)
        requested = inherited

    raw_grants = cfg.get("capability_grants")
    default_name = str(cfg.get("default_capability_grant") or "").strip()
    if raw_grants in (None, {}):
        if requested:
            raise ValueError(
                f"Unknown delegation capability_grant {requested!r}: no "
                "capability grants are configured."
            )
        if default_name:
            raise ValueError(
                "delegation.default_capability_grant is set but "
                "delegation.capability_grants is empty"
            )
        return None, None
    if not isinstance(raw_grants, dict):
        raise ValueError("delegation.capability_grants must be a mapping")

    selected_name = requested or default_name
    if not selected_name:
        return None, None
    raw = raw_grants.get(selected_name)
    if not isinstance(raw, dict):
        choices = ", ".join(configured_capability_grant_names(cfg)) or "<none>"
        raise ValueError(
            f"Unknown delegation capability_grant {selected_name!r}. "
            f"Allowed grants: {choices}."
        )
    return selected_name, _normalize_grant(selected_name, raw)


def bind_capability_grant(
    grant: Optional[dict[str, Any]], goal: str
) -> Optional[dict[str, Any]]:
    if not isinstance(grant, dict):
        return None
    bound = copy.deepcopy(grant)
    material = f"{bound.get('grant_digest', '')}\n{goal.strip()}"
    bound["mission_grant_id"] = "cap-" + hashlib.sha256(
        material.encode("utf-8")
    ).hexdigest()[:20]
    return bound


def capability_max_iterations(grant: Optional[dict[str, Any]], configured: int) -> int:
    grant_limit = int((grant or {}).get("max_iterations") or 0)
    return min(configured, grant_limit) if grant_limit else configured


def capability_child_timeout(
    grant: Optional[dict[str, Any]], configured: Optional[float]
) -> Optional[float]:
    grant_limit = float((grant or {}).get("child_timeout_seconds") or 0)
    if not grant_limit:
        return configured
    if configured is None or configured <= 0:
        return grant_limit
    return min(float(configured), grant_limit)


def capability_prompt_block(grant: Optional[dict[str, Any]]) -> str:
    if not isinstance(grant, dict):
        return ""
    bound_files = list(
        dict.fromkeys(
            path
            for spec in grant.get("controller_commands", [])
            for path in (spec.get("bound_files") or [])
        )
    )
    controllers = [
        " ".join(spec.get("argv") or [])
        + " [options: "
        + ", ".join(
            list(spec.get("flag_options") or [])
            + list(spec.get("value_options") or [])
            + list((spec.get("fixed_value_options") or {}).keys())
        )
        + "; bound_files: "
        + str(len(spec.get("bound_files") or []))
        + "]"
        for spec in grant.get("controller_commands", [])
    ]
    return (
        "\n\nMISSION CAPABILITY GRANT (operator-configured; inherited by nested workers):\n"
        f"- name: {grant.get('name')}\n"
        f"- mission_grant_id: {grant.get('mission_grant_id')}\n"
        f"- allowed_roots: {', '.join(grant.get('allowed_roots', [])) or 'none'}\n"
        f"- allowed_services: {', '.join(grant.get('allowed_services', [])) or 'none'}\n"
        f"- service_commands: {', '.join(grant.get('service_commands', [])) or 'none'}\n"
        f"- allowed_operations: {', '.join(grant.get('allowed_operations', [])) or 'none'}\n"
        f"- controller_commands: {', '.join(controllers) or 'none'}\n"
        f"- controller_bound_files: {', '.join(bound_files) or 'none'}\n"
        f"- mission_max_iterations: {grant.get('max_iterations') or 'delegation default'}\n"
        f"- mission_deadline_seconds: {grant.get('child_timeout_seconds') or 'delegation default'}\n"
        "- controller contract boundaries: "
        + ", ".join(grant.get("controller_contract_boundaries", []))
        + "\nUse only the exact plain-argv controllers and local service targets above. "
        "Controller/executable hashes are checked at authorization time. Values "
        "accepted by a typed option are inert controller input; the trusted "
        "controller remains responsible for its domain boundaries."
    )


def _plain_tokens(command: str) -> list[str] | None:
    if not isinstance(command, str) or not command.strip():
        return None
    if _has_active_shell_meta(command):
        return None
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    return tokens or None


def _has_active_shell_meta(command: str) -> bool:
    """Return True for shell syntax that remains active outside single quotes."""
    single_quoted = False
    double_quoted = False
    escaped = False
    unquoted_meta = set("$`;&|<>{}()[]*?\n\r")
    for char in command:
        if escaped:
            escaped = False
            continue
        if char == "\\" and not single_quoted:
            # Alternate escaped spellings complicate exact argv matching even
            # when the escaped byte would otherwise be harmless.
            return True
        if char == "'" and not double_quoted:
            single_quoted = not single_quoted
            continue
        if char == '"' and not single_quoted:
            double_quoted = not double_quoted
            continue
        if single_quoted:
            continue
        if char in {"$", "`", "\n", "\r"}:
            return True
        if not double_quoted and char in unquoted_meta:
            return True
    return single_quoted or double_quoted or escaped


def _interactive_program_invocation(tokens: list[str]) -> bool:
    """Identify shell/REPL launches that can accept unguarded process stdin."""
    if not tokens:
        return False
    program = Path(tokens[0]).name.lower()
    args = tokens[1:]
    if program in _INTERACTIVE_SHELLS:
        if args and all(arg in {"--help", "--version", "-h"} for arg in args):
            return False
        if any(
            arg in {"-c", "--command"}
            or (arg.startswith("-") and not arg.startswith("--") and "c" in arg[1:])
            for arg in args
        ):
            return False
        return not any(not arg.startswith("-") for arg in args)
    if program in _INTERACTIVE_REPLS:
        if any(arg in {"-i", "--interactive"} for arg in args):
            return True
        if args and all(
            arg in {"--help", "--version", "-h", "-v", "-V"} for arg in args
        ):
            return False
        if any(arg in {"-c", "-e", "--eval", "-m", "-p", "--print"} for arg in args):
            return False
        return not any(not arg.startswith("-") for arg in args)
    return False


def _protected_token(token: str) -> bool:
    lowered = token.lower()
    fragments = re.split(r"[/=:]", lowered)
    for fragment in fragments:
        if not fragment:
            continue
        if fragment in _PROTECTED_COMPONENTS or re.fullmatch(
            r"\.env(?:$|[._-].+)", fragment
        ):
            return True
        if fragment.endswith((".pem", ".key")):
            return True
        if re.fullmatch(
            r"(?:credentials?|secrets?|private[-_]?key|api[-_]?key|"
            r"access[-_]?token|refresh[-_]?token|password)(?:\..+)?",
            fragment,
        ):
            return True
    return False


def _cwd_allowed(cwd: Optional[str], roots: list[str]) -> bool:
    if not cwd:
        return False
    path = Path(os.path.expandvars(os.path.expanduser(cwd)))
    return path.is_absolute() and _inside_root(path, roots)


def _controller_sensitive(
    tokens: list[str],
    grant: Optional[dict[str, Any]],
    *,
    effective_cwd: Optional[str] = None,
) -> bool:
    if not isinstance(grant, dict):
        return False
    for identity in grant.get("controller_identities", []):
        argv = identity.get("argv") or []
        if argv and tokens[: len(argv)] == argv:
            return True
        identity_path = str(identity.get("identity_path") or "")
        identity_command_path = str(identity.get("identity_command_path") or "")
        controller_paths = {
            identity_path,
            identity_command_path,
        }
        native_controller = identity_command_path == str(
            identity.get("executable_command_path") or ""
        )
        if native_controller:
            controller_paths.update(
                {
                    str(identity.get("executable_command_path") or ""),
                    str(identity.get("executable_path") or ""),
                }
            )
        if any(token in controller_paths for token in tokens if token):
            return True
        # Catch wrapper/symlink spellings of the controller itself without
        # treating a shared interpreter as controller-sensitive.
        for token in tokens:
            candidate = Path(token)
            try:
                if candidate.is_absolute():
                    resolved_candidate = str(candidate.resolve(strict=True))
                elif effective_cwd:
                    resolved_candidate = str(
                        (Path(effective_cwd) / candidate).resolve(strict=True)
                    )
                else:
                    continue
            except OSError:
                continue
            if resolved_candidate == identity_path:
                return True
    return False


def _controller_shell_pattern_hint(
    command: str,
    grant: Optional[dict[str, Any]],
    *,
    effective_cwd: Optional[str] = None,
) -> bool:
    """Recognize shell-glob spellings that can expand to a bound controller."""
    if not isinstance(grant, dict) or not _has_active_shell_meta(command):
        return False
    try:
        shell_tokens = shlex.split(command)
    except ValueError:
        return False
    patterns: list[str] = []
    for token in shell_tokens:
        if not any(char in token for char in "[*?"):
            continue
        candidate = Path(token)
        if candidate.is_absolute():
            patterns.append(str(candidate))
        elif effective_cwd:
            patterns.append(str(Path(effective_cwd) / candidate))
    if not patterns:
        return False
    for identity in grant.get("controller_identities", []):
        targets = {
            str(identity.get("identity_command_path") or ""),
            str(identity.get("identity_path") or ""),
        }
        if identity.get("identity_command_path") == identity.get(
            "executable_command_path"
        ):
            targets.update(
                {
                    str(identity.get("executable_command_path") or ""),
                    str(identity.get("executable_path") or ""),
                }
            )
        if any(
            target and fnmatch.fnmatchcase(target, pattern)
            for target in targets
            for pattern in patterns
        ):
            return True
    return False


def _controller_allowed(
    tokens: list[str],
    grant: dict[str, Any],
    *,
    effective_cwd: Optional[str],
) -> tuple[bool, str]:
    if "controller" not in grant.get("allowed_operations", []):
        return False, "controller_not_granted"
    if not _cwd_allowed(effective_cwd, list(grant.get("allowed_roots", []))):
        return False, "controller_cwd_outside_grant"
    for identity in grant.get("controller_identities", []):
        prefix = identity.get("argv") or []
        if tokens[: len(prefix)] != prefix:
            continue
        allowed, reason = _controller_args_allowed(
            tokens[len(prefix) :], identity.get("arg_schema") or {}
        )
        if not allowed:
            return False, reason
        # Credential-like values are never appropriate in mission argv. Other
        # values are inert data consumed by the content-bound controller and
        # are governed by its typed option schema, not shell semantics.
        if any(_protected_token(token) for token in tokens[len(prefix) :]):
            return False, "credential_or_secret_change"
        path = Path(str(identity.get("identity_command_path") or ""))
        executable = Path(str(identity.get("executable_command_path") or ""))
        try:
            resolved_executable = executable.resolve(strict=True)
            executable_digest = _sha256_file(resolved_executable)
            resolved = path.resolve(strict=True)
            digest = _sha256_file(resolved)
        except OSError:
            return False, "controller_identity_unavailable"
        if str(resolved_executable) != str(identity.get("executable_path")):
            return False, "controller_executable_changed"
        if executable_digest != identity.get("executable_sha256"):
            return False, "controller_executable_changed"
        if str(resolved) != str(identity.get("identity_path")):
            return False, "controller_identity_changed"
        if digest != identity.get("identity_sha256"):
            return False, "controller_content_changed"
        for dependency in identity.get("bound_file_identities", []):
            dependency_path = Path(str(dependency.get("command_path") or ""))
            try:
                resolved_dependency = dependency_path.resolve(strict=True)
                dependency_digest = _sha256_file(resolved_dependency)
            except OSError:
                return False, "controller_dependency_unavailable"
            if (
                str(resolved_dependency) != str(dependency.get("resolved_path"))
                or dependency_digest != dependency.get("sha256")
            ):
                return False, "controller_dependency_changed"
        return True, "controller"
    return False, "controller_argv_not_granted"


def _controller_args_allowed(
    args: list[str], schema: dict[str, Any]
) -> tuple[bool, str]:
    flags = set(schema.get("flag_options") or [])
    values = set(schema.get("value_options") or [])
    repeatable = set(schema.get("repeatable_value_options") or [])
    fixed = schema.get("fixed_value_options") or {}
    seen: set[str] = set()
    index = 0
    while index < len(args):
        option = args[index]
        if option in flags:
            if option in seen:
                return False, "controller_duplicate_option"
            seen.add(option)
            index += 1
            continue
        if option not in values and option not in fixed:
            return False, "controller_option_not_granted"
        if option in seen and option not in repeatable:
            return False, "controller_duplicate_option"
        if index + 1 >= len(args) or args[index + 1].startswith("--"):
            return False, "controller_option_value_missing"
        value = args[index + 1]
        if option in fixed and value not in set(fixed[option]):
            return False, "controller_option_value_not_granted"
        seen.add(option)
        index += 2
    return True, "controller_args"


def _service_lifecycle_index(
    tokens: list[str], *, effective_cwd: Optional[str] = None
) -> Optional[int]:
    for index, token in enumerate(tokens):
        program = Path(token).name.lower()
        candidate = Path(token)
        resolved_candidate: Optional[Path] = None
        try:
            if candidate.is_absolute():
                resolved_candidate = candidate.resolve(strict=True)
            elif (os.sep in token or (os.altsep and os.altsep in token)) and effective_cwd:
                resolved_candidate = (Path(effective_cwd) / candidate).resolve(strict=True)
            elif os.sep not in token:
                located = shutil.which(token)
                if located:
                    resolved_candidate = Path(located).resolve(strict=True)
        except OSError:
            resolved_candidate = None
        if resolved_candidate is not None:
            resolved_name = resolved_candidate.name.lower()
            if resolved_name in {"launchctl", "systemctl", "docker", "hermes"}:
                program = resolved_name
        tail = tokens[index + 1 :]
        if program == "launchctl" and tail and tail[0] in {
            "kickstart",
            "start",
            "stop",
            "bootout",
            "bootstrap",
            "load",
            "unload",
        }:
            return index
        if program == "systemctl" and tail and tail[0] in {
            "restart",
            "start",
            "stop",
            "reload",
            "enable",
            "disable",
        }:
            return index
        if program == "docker" and tail:
            if tail[0] in {"restart", "start", "stop", "kill", "rm"}:
                return index
            if len(tail) >= 2 and tail[0] == "compose" and tail[1] in {
                "restart",
                "up",
                "down",
                "stop",
                "start",
            }:
                return index
        if program == "hermes" and "gateway" in tail and any(
            verb in tail for verb in ("restart", "start", "stop")
        ):
            return index
    return None


def _service_reload_decision(
    tokens: list[str],
    grant: Optional[dict[str, Any]],
    *,
    effective_cwd: Optional[str],
    env_type: Optional[str],
) -> tuple[bool, str]:
    lifecycle_index = _service_lifecycle_index(tokens, effective_cwd=effective_cwd)
    if lifecycle_index is None:
        return False, "not_service_reload"
    if lifecycle_index != 0:
        return False, "service_wrapper_not_granted"
    if env_type != "local":
        return False, "capability_requires_local_terminal"
    if not isinstance(grant, dict):
        return False, "service_reload_not_granted"
    if not _cwd_allowed(effective_cwd, list(grant.get("allowed_roots", []))):
        return False, "service_cwd_outside_grant"

    program_token = tokens[0]
    program = Path(program_token).name.lower()
    if program_token != program:
        identity = next(
            (
                item
                for item in grant.get("service_identities", [])
                if item.get("command_path") == program_token
            ),
            None,
        )
    else:
        identity = None
    if identity is None:
        return False, "service_executable_not_granted"
    path = Path(str(identity.get("command_path") or ""))
    try:
        resolved = path.resolve(strict=True)
        digest = _sha256_file(resolved)
    except OSError:
        return False, "service_executable_unavailable"
    if (
        str(resolved) != str(identity.get("resolved_path"))
        or digest != identity.get("sha256")
    ):
        return False, "service_executable_changed"
    if "service_reload" not in grant.get("allowed_operations", []):
        return False, "service_reload_not_granted"
    services = set(grant.get("allowed_services", []))

    if program == "launchctl":
        if tokens[1:3] == ["kickstart", "-k"] and len(tokens) == 4:
            target = tokens[3]
        elif len(tokens) == 3 and tokens[1] == "kickstart":
            target = tokens[2]
        else:
            return False, "service_reload_argv_not_granted"
        getuid = getattr(os, "getuid", None)
        if not callable(getuid):
            return False, "service_manager_unavailable"
        match = re.fullmatch(rf"gui/{getuid()}/([A-Za-z0-9_.:@+-]+)", target)
        if not match or match.group(1) not in services:
            return False, "service_target_not_granted"
        return True, "service_reload"

    if program == "systemctl":
        if len(tokens) < 3 or tokens[1] != "restart" or any(
            token.startswith("-") for token in tokens[2:]
        ):
            return False, "service_reload_argv_not_granted"
        selected = tokens[2:]
        return (
            (True, "service_reload")
            if selected and all(service in services for service in selected)
            else (False, "service_target_not_granted")
        )

    # Docker/compose and Hermes gateway lifecycle remain intentionally
    # ungrantable here. Docker project selection is environment/cwd-sensitive,
    # and a Hermes worker can kill the process that owns its own mission.
    return False, "service_manager_not_grantable"


def evaluate_capability_command(
    command: str,
    grant: Optional[dict[str, Any]],
    *,
    effective_cwd: Optional[str] = None,
    env_type: Optional[str] = None,
) -> dict[str, Any]:
    """Return a typed allow/deny/pass decision for terminal guard integration."""
    tokens = _plain_tokens(command)
    raw_lower = command.lower() if isinstance(command, str) else ""
    service_hint = bool(_SERVICE_LIFECYCLE_HINT_RE.search(raw_lower)) or bool(
        _SERVICE_MANAGER_HINT_RE.search(raw_lower)
        and _SERVICE_MUTATION_HINT_RE.search(raw_lower)
    )
    # If active shell syntax and a lifecycle mutation verb coexist, classify
    # the command as service-sensitive even when globbing/variables obscure the
    # manager spelling. Exact granted service commands never need shell syntax.
    service_hint = service_hint or bool(
        isinstance(command, str)
        and _has_active_shell_meta(command)
        and _SERVICE_MUTATION_HINT_RE.search(raw_lower)
    )
    controller_hint = False
    if isinstance(grant, dict):
        controller_hint = any(
            str(identity.get("identity_path") or "") in command
            or " ".join(identity.get("argv") or []) in command
            for identity in grant.get("controller_identities", [])
        )
        controller_hint = controller_hint or _controller_shell_pattern_hint(
            command,
            grant,
            effective_cwd=effective_cwd,
        )
    if not tokens:
        if service_hint or controller_hint:
            return {
                "matched": True,
                "allowed": False,
                "reason": "capability_requires_plain_argv",
            }
        return {"matched": False, "allowed": False, "reason": "not_capability_sensitive"}

    if isinstance(grant, dict) and _interactive_program_invocation(tokens):
        return {
            "matched": True,
            "allowed": False,
            "reason": "interactive_process_stdin_not_granted",
        }

    if _service_lifecycle_index(tokens, effective_cwd=effective_cwd) is not None:
        allowed, reason = _service_reload_decision(
            tokens,
            grant,
            effective_cwd=effective_cwd,
            env_type=env_type,
        )
        return {"matched": True, "allowed": allowed, "reason": reason}

    if service_hint:
        return {
            "matched": True,
            "allowed": False,
            "reason": "service_reload_argv_not_granted",
        }

    if _controller_sensitive(tokens, grant, effective_cwd=effective_cwd):
        if env_type != "local":
            return {
                "matched": True,
                "allowed": False,
                "reason": "capability_requires_local_terminal",
            }
        allowed, reason = _controller_allowed(
            tokens, grant or {}, effective_cwd=effective_cwd
        )
        return {"matched": True, "allowed": allowed, "reason": reason}

    return {"matched": False, "allowed": False, "reason": "not_capability_sensitive"}


def authorize_capability_command(
    command: str,
    grant: Optional[dict[str, Any]],
    *,
    workspace_path: Optional[str] = None,
    effective_cwd: Optional[str] = None,
    env_type: Optional[str] = "local",
) -> tuple[bool, str]:
    """Compatibility wrapper used by focused tests and diagnostics."""
    decision = evaluate_capability_command(
        command,
        grant,
        effective_cwd=effective_cwd or workspace_path,
        env_type=env_type,
    )
    return bool(decision["allowed"]), str(decision["reason"])
