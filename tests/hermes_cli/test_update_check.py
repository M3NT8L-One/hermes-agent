"""Tests for the update check mechanism in hermes_cli.banner."""

import json
import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_version_string_no_v_prefix():
    """__version__ should be bare semver without a 'v' prefix."""
    from hermes_cli import __version__
    assert not __version__.startswith("v"), f"__version__ should not start with 'v', got {__version__!r}"


def test_check_for_updates_uses_cache(tmp_path, monkeypatch):
    """When cache is fresh, check_for_updates should return cached value without calling git."""
    import hermes_cli.banner as banner
    from hermes_cli import __version__

    # Create a fake git repo and fresh cache
    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()
    monkeypatch.setattr(banner, "__file__", str(repo_dir / "hermes_cli" / "banner.py"))

    identity = {
        "mode": "git",
        "head": "local-sha",
        "remote": "upstream",
        "tracking_ref": "upstream/main",
        "tracking_tip": "official-sha",
        "remote_fingerprint": "remote-digest",
    }

    cache_file = tmp_path / ".update_check"
    cache_file.write_text(json.dumps({
        "ts": time.time(),
        "behind": 3,
        "ver": __version__,
        "policy": banner._UPDATE_CHECK_CACHE_POLICY,
        "identity": identity,
    }))

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with patch(
        "hermes_cli.banner._local_git_cache_identity", return_value=identity
    ), patch("hermes_cli.banner._check_via_local_git") as mock_check:
        result = banner.check_for_updates()

    assert result == 3
    mock_check.assert_not_called()


def test_check_for_updates_invalidates_on_version_change(tmp_path, monkeypatch):
    """A fresh cache from a different installed version must be re-checked, not reused.

    Regression for #34491: after `pip install --upgrade`, VERSION changes but the
    cache's 6h TTL hadn't expired and rev was unchanged (both None), so the stale
    'behind' count survived the upgrade. The version guard forces a recheck.
    """
    import hermes_cli.banner as banner

    # No local git checkout -> the PyPI path is exercised (pip-install class).
    fake_banner = tmp_path / "hermes_cli" / "banner.py"
    fake_banner.parent.mkdir(parents=True, exist_ok=True)
    fake_banner.touch()
    monkeypatch.setattr(banner, "__file__", str(fake_banner))

    # Fresh (within TTL) cache that says "behind", but stamped with an OLD version.
    cache_file = tmp_path / ".update_check"
    cache_file.write_text(
        json.dumps({"ts": time.time(), "behind": 1, "rev": None, "ver": "0.0.1-old"})
    )

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_REVISION", raising=False)
    with patch("hermes_cli.banner.subprocess.run") as mock_run:
        result = banner.check_for_updates()

    # Stale-version cache rejected -> fresh check ran. No git checkout and no
    # embedded rev means we can't determine update status, so result is None.
    assert result is None
    mock_run.assert_not_called()

    # Cache rewritten with the current installed version.
    written = json.loads(cache_file.read_text())
    assert written["ver"] == banner.VERSION


def test_check_for_updates_expired_cache(tmp_path, monkeypatch):
    """When cache is expired, check_for_updates should call git fetch."""
    from hermes_cli.banner import check_for_updates

    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    # Write an expired cache (timestamp far in the past)
    cache_file = tmp_path / ".update_check"
    cache_file.write_text(json.dumps({"ts": 0, "behind": 1}))

    def fake_run(cmd, **kwargs):
        del kwargs
        if cmd == ["git", "remote", "get-url", "origin"]:
            return MagicMock(
                returncode=0,
                stdout="https://github.com/NousResearch/hermes-agent.git\n",
            )
        return MagicMock(returncode=0, stdout="5\n")

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with patch("hermes_cli.banner.subprocess.run", side_effect=fake_run) as mock_run:
        result = check_for_updates()

    assert result == 5
    commands = [call.args[0] for call in mock_run.call_args_list]
    assert ["git", "fetch", "origin", "--quiet"] in commands
    assert ["git", "rev-list", "--count", "HEAD..origin/main"] in commands


def _git_identity(**updates):
    identity = {
        "mode": "git",
        "head": "local-old",
        "remote": "upstream",
        "tracking_ref": "upstream/main",
        "tracking_tip": "official-old",
        "remote_fingerprint": "remote-old",
    }
    identity.update(updates)
    return identity


@pytest.mark.parametrize(
    "changed_identity",
    [
        _git_identity(head="local-new"),
        _git_identity(tracking_tip="official-new"),
        _git_identity(
            remote="origin",
            tracking_ref="origin/main",
            remote_fingerprint="remote-new",
        ),
    ],
    ids=["head-moved", "tracking-tip-moved", "canonical-remote-changed"],
)
def test_check_for_updates_invalidates_cache_when_git_identity_changes(
    tmp_path, monkeypatch, changed_identity
):
    """A merge, fetch, or canonical-remote change must defeat the six-hour TTL."""
    import hermes_cli.banner as banner

    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()
    monkeypatch.setattr(banner, "__file__", str(repo_dir / "hermes_cli" / "banner.py"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cache_file = tmp_path / ".update_check"
    cache_file.write_text(json.dumps({
        "ts": time.time(),
        "behind": 13,
        "rev": None,
        "ver": banner.VERSION,
        "policy": banner._UPDATE_CHECK_CACHE_POLICY,
        "identity": _git_identity(),
    }))

    with patch(
        "hermes_cli.banner._local_git_cache_identity",
        side_effect=[changed_identity, changed_identity],
    ), patch(
        "hermes_cli.banner._check_via_local_git", return_value=0
    ) as mock_check:
        result = banner.check_for_updates()

    assert result == 0
    mock_check.assert_called_once_with(repo_dir)
    written = json.loads(cache_file.read_text())
    assert written["behind"] == 0
    assert written["identity"] == changed_identity


def test_check_for_updates_stores_post_fetch_git_identity(tmp_path, monkeypatch):
    """The cache records the tracking tip after update-check fetch moves it."""
    import hermes_cli.banner as banner

    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()
    monkeypatch.setattr(banner, "__file__", str(repo_dir / "hermes_cli" / "banner.py"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    before = _git_identity(tracking_tip="official-before-fetch")
    after = _git_identity(tracking_tip="official-after-fetch")

    with patch(
        "hermes_cli.banner._local_git_cache_identity", side_effect=[before, after]
    ), patch("hermes_cli.banner._check_via_local_git", return_value=2):
        result = banner.check_for_updates()

    assert result == 2
    written = json.loads((tmp_path / ".update_check").read_text())
    assert written["identity"] == after


def test_local_git_cache_identity_uses_official_upstream_without_storing_url(tmp_path):
    import hermes_cli.banner as banner

    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()

    def fake_run(cmd, **kwargs):
        del kwargs
        outputs = {
            ("git", "remote", "get-url", "origin"):
                "https://github.com/private-owner/hermes-agent.git\n",
            ("git", "remote", "get-url", "upstream"):
                "https://github.com/NousResearch/hermes-agent.git\n",
            ("git", "rev-parse", "HEAD"): "local-sha\n",
            ("git", "rev-parse", "upstream/main"): "official-sha\n",
        }
        return MagicMock(returncode=0, stdout=outputs[tuple(cmd)])

    with patch("hermes_cli.banner.subprocess.run", side_effect=fake_run):
        identity = banner._local_git_cache_identity(repo_dir)

    assert identity == {
        "mode": "git",
        "head": "local-sha",
        "remote": "upstream",
        "tracking_ref": "upstream/main",
        "tracking_tip": "official-sha",
        "remote_fingerprint": (
            "8df3514a52f5d74372a33c5c53190620"
            "464da3b05751ae1ca76187d1ac0b4ffb"
        ),
    }
    assert "github.com" not in json.dumps(identity)


def test_check_for_updates_official_ssh_origin_uses_https_probe(tmp_path):
    """Passive update checks must not trigger SSH auth for official installs."""
    import hermes_cli.banner as banner

    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd == ["git", "remote", "get-url", "origin"]:
            return MagicMock(returncode=0, stdout="git@github.com:NousResearch/hermes-agent.git\n")
        if cmd == ["git", "rev-parse", "HEAD"]:
            return MagicMock(returncode=0, stdout="local-sha\n")
        if cmd == [
            "git",
            "ls-remote",
            "https://github.com/NousResearch/hermes-agent.git",
            "refs/heads/main",
        ]:
            return MagicMock(returncode=0, stdout="upstream-sha\trefs/heads/main\n")
        raise AssertionError(f"unexpected git command: {cmd!r}")

    with patch("hermes_cli.banner.subprocess.run", side_effect=fake_run):
        result = banner._check_via_local_git(repo_dir)

    assert result == 1
    assert ["git", "fetch", "origin", "--quiet"] not in calls


def test_check_via_local_git_shallow_clone_behind_reports_no_count(tmp_path):
    """Shallow installer clones must report presence-only, never a bogus count.

    On a ``git clone --depth 1`` checkout the history stops at one commit, so
    counting ``HEAD..origin/main`` across the shallow boundary yields a huge
    nonsense number (the "12492 commits behind" banner). The shallow path must
    compare tip SHAs and return UPDATE_AVAILABLE_NO_COUNT instead, and must
    never run ``git rev-list --count``.
    """
    import hermes_cli.banner as banner

    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd == ["git", "remote", "get-url", "origin"]:
            return MagicMock(returncode=0, stdout="https://github.com/NousResearch/hermes-agent.git\n")
        if cmd == ["git", "rev-parse", "--is-shallow-repository"]:
            return MagicMock(returncode=0, stdout="true\n")
        if cmd[:2] == ["git", "fetch"]:
            return MagicMock(returncode=0, stdout="")
        if cmd == ["git", "rev-parse", "HEAD"]:
            return MagicMock(returncode=0, stdout="local-sha\n")
        if cmd == ["git", "rev-parse", "FETCH_HEAD"]:
            return MagicMock(returncode=0, stdout="upstream-sha\n")
        if cmd[:3] == ["git", "rev-list", "--count"]:
            raise AssertionError("shallow path must not count across the boundary")
        raise AssertionError(f"unexpected git command: {cmd!r}")

    with patch("hermes_cli.banner.subprocess.run", side_effect=fake_run):
        result = banner._check_via_local_git(repo_dir)

    assert result == banner.UPDATE_AVAILABLE_NO_COUNT
    # The shallow fetch must preserve the boundary (--depth 1), not unshallow.
    assert ["git", "fetch", "origin", "--depth", "1", "--quiet"] in calls


def test_check_via_local_git_shallow_clone_up_to_date(tmp_path):
    """Shallow clone whose tip matches upstream reports up-to-date (0)."""
    import hermes_cli.banner as banner

    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    def fake_run(cmd, **kwargs):
        if cmd == ["git", "remote", "get-url", "origin"]:
            return MagicMock(returncode=0, stdout="https://github.com/NousResearch/hermes-agent.git\n")
        if cmd == ["git", "rev-parse", "--is-shallow-repository"]:
            return MagicMock(returncode=0, stdout="true\n")
        if cmd[:2] == ["git", "fetch"]:
            return MagicMock(returncode=0, stdout="")
        if cmd == ["git", "rev-parse", "HEAD"]:
            return MagicMock(returncode=0, stdout="same-sha\n")
        if cmd == ["git", "rev-parse", "FETCH_HEAD"]:
            return MagicMock(returncode=0, stdout="same-sha\n")
        raise AssertionError(f"unexpected git command: {cmd!r}")

    with patch("hermes_cli.banner.subprocess.run", side_effect=fake_run):
        result = banner._check_via_local_git(repo_dir)

    assert result == 0


def test_check_via_local_git_full_clone_keeps_exact_count(tmp_path):
    """Full (non-shallow) clones keep the exact rev-list count path."""
    import hermes_cli.banner as banner

    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    def fake_run(cmd, **kwargs):
        if cmd == ["git", "remote", "get-url", "origin"]:
            return MagicMock(returncode=0, stdout="https://github.com/NousResearch/hermes-agent.git\n")
        if cmd == ["git", "rev-parse", "--is-shallow-repository"]:
            return MagicMock(returncode=0, stdout="false\n")
        if cmd[:2] == ["git", "fetch"]:
            return MagicMock(returncode=0, stdout="")
        if cmd[:3] == ["git", "rev-list", "--count"]:
            return MagicMock(returncode=0, stdout="7\n")
        raise AssertionError(f"unexpected git command: {cmd!r}")

    with patch("hermes_cli.banner.subprocess.run", side_effect=fake_run):
        result = banner._check_via_local_git(repo_dir)

    assert result == 7


def test_check_via_local_git_private_fork_uses_official_upstream(tmp_path):
    """A clean private origin must not hide drift from official upstream."""
    import hermes_cli.banner as banner

    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()
    calls = []

    def fake_run(cmd, **kwargs):
        del kwargs
        calls.append(cmd)
        if cmd == ["git", "remote", "get-url", "origin"]:
            return MagicMock(
                returncode=0,
                stdout="https://github.com/M3NT8L-One/hermes-agent.git\n",
            )
        if cmd == ["git", "remote", "get-url", "upstream"]:
            return MagicMock(
                returncode=0,
                stdout="https://github.com/NousResearch/hermes-agent.git\n",
            )
        if cmd == ["git", "rev-parse", "--is-shallow-repository"]:
            return MagicMock(returncode=0, stdout="false\n")
        if cmd[:3] == ["git", "fetch", "upstream"]:
            return MagicMock(returncode=0, stdout="")
        if cmd == ["git", "rev-list", "--count", "HEAD..upstream/main"]:
            return MagicMock(returncode=0, stdout="4\n")
        raise AssertionError(f"unexpected git command: {cmd!r}")

    with patch("hermes_cli.banner.subprocess.run", side_effect=fake_run):
        result = banner._check_via_local_git(repo_dir)

    assert result == 4
    assert ["git", "fetch", "upstream", "--quiet"] in calls


def test_check_for_updates_no_git_dir(tmp_path, monkeypatch):
    """Returns None when .git directory doesn't exist anywhere (no source tree)."""
    import hermes_cli.banner as banner

    # Create a fake banner.py so the fallback path also has no .git
    fake_banner = tmp_path / "hermes_cli" / "banner.py"
    fake_banner.parent.mkdir(parents=True, exist_ok=True)
    fake_banner.touch()

    monkeypatch.setattr(banner, "__file__", str(fake_banner))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with patch("hermes_cli.banner.subprocess.run") as mock_run:
        result = banner.check_for_updates()
    assert result is None
    mock_run.assert_not_called()


def test_check_for_updates_fallback_to_project_root(tmp_path, monkeypatch):
    """Dev install: falls back to Path(__file__).parent.parent when HERMES_HOME has no git repo."""
    import hermes_cli.banner as banner

    project_root = Path(banner.__file__).parent.parent.resolve()
    if not (project_root / ".git").exists():
        pytest.skip("Not running from a git checkout")

    # Point HERMES_HOME at a temp dir with no hermes-agent/.git
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with patch("hermes_cli.banner.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="0\n")
        result = banner.check_for_updates()
    # Should have fallen back to project root and run git commands
    assert mock_run.call_count >= 1


def test_check_for_updates_docker_returns_none(tmp_path, monkeypatch):
    """Inside the Docker image, check_for_updates() must short-circuit to None.

    Regression: the published image excludes .git (.dockerignore) and sets no
    HERMES_REVISION (nix-only), so without a docker guard check_for_updates()
    would fall through and try to probe a non-existent git checkout. The guard
    must return None (so the > 0 render guards stay false) AND not reach the
    git probe or write a cache entry.
    """
    import hermes_cli.banner as banner

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cache_file = tmp_path / ".update_check"

    with patch("hermes_cli.config.detect_install_method", return_value="docker"), \
         patch("hermes_cli.banner.subprocess.run") as mock_run:
        result = banner.check_for_updates()

    assert result is None
    # The git probe should not have run.
    mock_run.assert_not_called()
    # And no phantom "behind" count should be cached for the next 6h.
    assert not cache_file.exists()


def test_prefetch_non_blocking():
    """prefetch_update_check() should return immediately without blocking."""
    import hermes_cli.banner as banner

    # Reset module state
    banner._update_result = None
    banner._update_check_done = threading.Event()

    with patch.object(banner, "check_for_updates", return_value=5):
        start = time.monotonic()
        banner.prefetch_update_check()
        elapsed = time.monotonic() - start

        # Should return almost immediately (well under 1 second)
        assert elapsed < 1.0

        # Wait for the background thread to finish
        banner._update_check_done.wait(timeout=5)
        assert banner._update_result == 5


def test_invalidate_update_cache_clears_all_profiles(tmp_path):
    """_invalidate_update_cache() should delete .update_check from ALL profiles."""
    from hermes_cli.main import _invalidate_update_cache

    # Build a fake ~/.hermes with default + two named profiles
    default_home = tmp_path / ".hermes"
    default_home.mkdir()
    (default_home / ".update_check").write_text('{"ts":1,"behind":50}')

    profiles_root = default_home / "profiles"
    for name in ("ops", "dev"):
        p = profiles_root / name
        p.mkdir(parents=True)
        (p / ".update_check").write_text('{"ts":1,"behind":50}')

    with patch.object(Path, "home", return_value=tmp_path), \
         patch.dict(os.environ, {"HERMES_HOME": str(default_home)}):
        _invalidate_update_cache()

    # All three caches should be gone
    assert not (default_home / ".update_check").exists(), "default profile cache not cleared"
    assert not (profiles_root / "ops" / ".update_check").exists(), "ops profile cache not cleared"
    assert not (profiles_root / "dev" / ".update_check").exists(), "dev profile cache not cleared"


def test_invalidate_update_cache_no_profiles_dir(tmp_path):
    """Works fine when no profiles directory exists (single-profile setup)."""
    from hermes_cli.main import _invalidate_update_cache

    default_home = tmp_path / ".hermes"
    default_home.mkdir()
    (default_home / ".update_check").write_text('{"ts":1,"behind":5}')

    with patch.object(Path, "home", return_value=tmp_path), \
         patch.dict(os.environ, {"HERMES_HOME": str(default_home)}):
        _invalidate_update_cache()

    assert not (default_home / ".update_check").exists()
