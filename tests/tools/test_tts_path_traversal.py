"""Regression: text_to_speech_tool output_path must reject '..' traversal.

The TTS surface accepts agent/user-supplied absolute paths (writing to a
chosen file is the whole point). What it must reject is paths that use
``..`` components to escape their declared base — those are almost
always either a bug or prompt-injection-controlled
(e.g. ``output_path="audio/../../etc/cron.d/x"``).
"""

import json
from pathlib import Path

from tools.tts_tool import text_to_speech_tool


def _stub_edge_backend(monkeypatch):
    """Exercise the path guard without making a real synthesis request."""

    async def generate(_text, output_path, _config):
        Path(output_path).write_bytes(b"test-audio")

    monkeypatch.setattr("tools.tts_tool._load_tts_config", lambda: {})
    monkeypatch.setattr("tools.tts_tool._import_edge_tts", lambda: object())
    monkeypatch.setattr("tools.tts_tool._generate_edge_tts", generate)


def test_output_path_rejects_traversal_escape():
    """A path with '..' components must be rejected before any provider work."""
    result = json.loads(text_to_speech_tool(
        text="hello",
        output_path="audio/../../etc/cron.d/malicious",
    ))
    assert result["success"] is False
    assert "traversal" in result["error"].lower()


def test_output_path_rejects_bare_dotdot():
    """Bare '..' prefix must be rejected."""
    result = json.loads(text_to_speech_tool(
        text="hello",
        output_path="../escape.mp3",
    ))
    assert result["success"] is False
    assert "traversal" in result["error"].lower()


def test_output_path_absolute_path_passes_guard(tmp_path, monkeypatch):
    """Explicit absolute paths must pass the traversal guard.

    The agent legitimately writes audio to user-specified absolute paths;
    only ``..`` components are rejected. Any subsequent failure (no
    provider configured, etc.) is fine — the assertion is specifically
    that the 'traversal' rejection didn't fire.
    """
    _stub_edge_backend(monkeypatch)
    inside = tmp_path / "clip.mp3"
    result = json.loads(text_to_speech_tool(
        text="hello",
        output_path=str(inside),
    ))
    error = result.get("error", "")
    assert "traversal" not in error.lower()


def test_output_path_relative_no_dotdot_passes_guard(tmp_path, monkeypatch):
    """Relative paths without '..' components must pass the guard."""
    _stub_edge_backend(monkeypatch)
    monkeypatch.chdir(tmp_path)
    result = json.loads(text_to_speech_tool(
        text="hello",
        output_path="subdir/clip.mp3",
    ))
    error = result.get("error", "")
    assert "traversal" not in error.lower()


def test_output_path_rejects_hermes_oauth_store(tmp_path, monkeypatch):
    """TTS output_path must not bypass the shared protected-file write guard."""
    import agent.file_safety as file_safety

    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    monkeypatch.setattr(file_safety, "_hermes_home_path", lambda: hermes_home)
    monkeypatch.setattr(file_safety, "_hermes_root_path", lambda: hermes_home)

    target = hermes_home / ".anthropic_oauth.json"
    result = json.loads(text_to_speech_tool(
        text="hello",
        output_path=str(target),
    ))

    assert result["success"] is False
    assert "protected credential" in result["error"]
    assert not target.exists()


def test_output_path_rejects_mcp_token_directory(tmp_path, monkeypatch):
    """TTS output_path must not write synthesized audio over MCP token files."""
    import agent.file_safety as file_safety

    hermes_home = tmp_path / "hermes-home"
    token_dir = hermes_home / "mcp-tokens"
    token_dir.mkdir(parents=True)
    monkeypatch.setattr(file_safety, "_hermes_home_path", lambda: hermes_home)
    monkeypatch.setattr(file_safety, "_hermes_root_path", lambda: hermes_home)

    target = token_dir / "server.mp3"
    result = json.loads(text_to_speech_tool(
        text="hello",
        output_path=str(target),
    ))

    assert result["success"] is False
    assert "protected credential" in result["error"]
    assert not target.exists()
