"""scripts/commit-daily.sh against a local stand-in for GitHub (a bare repo):
the record reaches main even when main moved during the run, and the job's
own checkout never moves, so later steps keep running the code that started."""

import subprocess
from pathlib import Path

import pytest

SCRIPT = Path("scripts/commit-daily.sh").resolve()


def _git(cwd: Path, *args: str) -> str:
    got = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    return got.stdout.strip()


@pytest.fixture
def repos(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    """origin (bare, on main), a second clone that moves main, and the job's clone."""
    empty = tmp_path / "gitconfig"
    empty.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for k in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{k}_NAME", "test")
        monkeypatch.setenv(f"GIT_{k}_EMAIL", "test@example.invalid")
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "-q", "--bare", "-b", "main", str(origin))
    other = tmp_path / "other"
    _git(tmp_path, "clone", "-q", str(origin), str(other))
    (other / "code.py").write_text("v1\n")
    (other / "runs" / "daily").mkdir(parents=True)
    (other / "runs" / "daily" / "2026-11-05.json").write_text("{}")
    _git(other, "add", ".")
    _git(other, "commit", "-q", "-m", "start")
    _git(other, "push", "-q", "origin", "HEAD:main")
    job = tmp_path / "job"
    _git(tmp_path, "clone", "-q", str(origin), str(job))
    monkeypatch.setenv("TENQ_PUSH_URL", str(origin))
    return origin, other, job


def _save(job: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), *args], cwd=job, capture_output=True, text=True, check=False
    )


def _on_main(origin: Path, path: str) -> str:
    return _git(origin, "show", f"main:{path}")


def test_the_record_reaches_main_and_the_job_stays_on_its_code(
    repos: tuple[Path, Path, Path],
) -> None:
    origin, other, job = repos
    started = _git(job, "rev-parse", "HEAD")
    # a pull request merges while the job runs
    (other / "code.py").write_text("v2\n")
    _git(other, "commit", "-qam", "newer code")
    _git(other, "push", "-q", "origin", "HEAD:main")
    record = job / "runs" / "daily" / "2026-11-06.json"
    record.write_text('{"email": {"status": "sending"}}')
    (job / "runs" / "daily" / ".2026-11-06.json.tmp").write_text("half")

    got = _save(job)
    assert got.returncode == 0, got.stderr
    assert _on_main(origin, "runs/daily/2026-11-06.json") == '{"email": {"status": "sending"}}'
    assert _on_main(origin, "code.py") == "v2"  # the merge is kept, not undone
    assert "runs/daily/.2026-11-06.json.tmp" not in _git(origin, "ls-tree", "-r", "main")
    # the job's checkout did not move: it still runs the code it started with
    assert _git(job, "rev-parse", "HEAD") == started
    assert (job / "code.py").read_text() == "v1\n"
    assert _git(job, "worktree", "list").count("\n") == 0  # the copy is cleaned up

    # the send step changes the record: the second save carries the change
    record.write_text('{"email": {"status": "sent"}}')
    assert _save(job).returncode == 0
    assert _on_main(origin, "runs/daily/2026-11-06.json") == '{"email": {"status": "sent"}}'
    assert _git(job, "rev-parse", "HEAD") == started

    # nothing new since: nothing to commit
    before = _git(origin, "rev-parse", "main")
    got = _save(job)
    assert got.returncode == 0 and "no new record" in got.stdout
    assert _git(origin, "rev-parse", "main") == before


def test_only_files_this_run_wrote_are_copied(repos: tuple[Path, Path, Path]) -> None:
    origin, other, job = repos
    # someone fixes an old record on main by hand during the run
    (other / "runs" / "daily" / "2026-11-05.json").write_text('{"fixed": true}')
    _git(other, "commit", "-qam", "hand fix")
    _git(other, "push", "-q", "origin", "HEAD:main")
    (job / "runs" / "daily" / "2026-11-06.json").write_text("{}")
    assert _save(job).returncode == 0
    # the job's stale copy of the old record does not undo the fix
    assert _on_main(origin, "runs/daily/2026-11-05.json") == '{"fixed": true}'


def test_a_consistency_record_is_saved_the_same_way(repos: tuple[Path, Path, Path]) -> None:
    origin, _, job = repos
    (job / "runs" / "consistency").mkdir()
    (job / "runs" / "consistency" / "2026-11-07-abc.json").write_text("{}")
    assert _save(job, "runs/consistency").returncode == 0
    assert _on_main(origin, "runs/consistency/2026-11-07-abc.json") == "{}"


def test_a_failed_push_fails_the_step(
    repos: tuple[Path, Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, job = repos
    (job / "runs" / "daily" / "2026-11-06.json").write_text("{}")
    monkeypatch.setenv("TENQ_PUSH_URL", str(tmp_path / "nowhere.git"))
    assert _save(job).returncode != 0


@pytest.mark.parametrize("workflow", ["daily", "consistency"])
def test_each_workflow_notes_the_commit_it_checked_out(workflow: str) -> None:
    import yaml

    flow = yaml.safe_load(Path(f".github/workflows/{workflow}.yml").read_text())
    (job,) = flow["jobs"].values()
    steps = job["steps"]
    assert steps[0]["uses"].startswith("actions/checkout") and steps[0]["with"]["ref"] == "main"
    # read from git right after the checkout, before any step can change it
    assert steps[1]["run"] == 'echo "TENQ_CODE_SHA=$(git rev-parse HEAD)" >> "$GITHUB_ENV"'
