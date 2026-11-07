"""scripts/commit-daily.sh against a local stand-in for GitHub (a bare repo):
the record reaches main even when main moved during the run, and the job's
own checkout never moves, so later steps keep running the code that started."""

import re
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
    monkeypatch.setenv("TENQ_RETRY_SLEEP", "0")
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


def test_main_unreachable_fails_the_step(
    repos: tuple[Path, Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, job = repos
    (job / "runs" / "daily" / "2026-11-06.json").write_text("{}")
    monkeypatch.setenv("TENQ_PUSH_URL", str(tmp_path / "nowhere.git"))
    got = _save(job)
    assert got.returncode == 1 and "could not fetch main after 2 tries" in got.stdout


def _hook(origin: Path, body: str) -> None:
    hook = origin / "hooks" / "pre-receive"
    hook.write_text("#!/bin/sh\n" + body + "\n")
    hook.chmod(0o755)


def test_a_refused_push_fails_the_step_and_main_keeps_its_tip(
    repos: tuple[Path, Path, Path],
) -> None:
    origin, _, job = repos
    _hook(origin, "exit 1")
    before = _git(origin, "rev-parse", "main")
    (job / "runs" / "daily" / "2026-11-06.json").write_text("{}")
    got = _save(job)
    assert got.returncode == 1 and "push failed after 2 tries" in got.stdout
    assert _git(origin, "rev-parse", "main") == before


def test_a_push_refused_once_is_tried_again(repos: tuple[Path, Path, Path]) -> None:
    origin, _, job = repos
    _hook(origin, "if [ ! -e refused-once ]; then touch refused-once; exit 1; fi")
    (job / "runs" / "daily" / "2026-11-06.json").write_text('{"a": 1}')
    got = _save(job)
    assert got.returncode == 0, got.stdout + got.stderr
    assert (origin / "refused-once").exists()
    assert _on_main(origin, "runs/daily/2026-11-06.json") == '{"a": 1}'


def test_a_record_another_run_saved_is_never_overwritten(
    repos: tuple[Path, Path, Path],
) -> None:
    origin, other, job = repos
    # another run claimed the same day after this job checked out
    (other / "runs" / "daily" / "2026-11-06.json").write_text('{"claim": "A"}')
    _git(other, "add", ".")
    _git(other, "commit", "-qm", "the other run's claim")
    _git(other, "push", "-q", "origin", "HEAD:main")
    (job / "runs" / "daily" / "2026-11-06.json").write_text('{"claim": "B"}')
    got = _save(job)
    assert got.returncode == 1 and "main changed runs/daily/2026-11-06.json" in got.stdout
    assert _on_main(origin, "runs/daily/2026-11-06.json") == '{"claim": "A"}'


def test_a_change_on_main_between_this_runs_saves_stops_the_second(
    repos: tuple[Path, Path, Path],
) -> None:
    origin, other, job = repos
    record = job / "runs" / "daily" / "2026-11-06.json"
    record.write_text('{"email": {"status": "sending"}}')
    assert _save(job).returncode == 0  # the claim
    # someone edits the claimed record on main before the send step saves
    _git(other, "pull", "-q", "origin", "main")
    (other / "runs" / "daily" / "2026-11-06.json").write_text('{"by": "hand"}')
    _git(other, "commit", "-qam", "hand edit")
    _git(other, "push", "-q", "origin", "HEAD:main")
    record.write_text('{"email": {"status": "sent"}}')
    assert _save(job).returncode == 1
    assert _on_main(origin, "runs/daily/2026-11-06.json") == '{"by": "hand"}'


def test_odd_file_names_are_saved_too(repos: tuple[Path, Path, Path]) -> None:
    origin, _, job = repos
    for name in ('q"uote.json', "\u00e9.json", "with space.json"):
        (job / "runs" / "daily" / name).write_text("{}")
    assert _save(job).returncode == 0
    listed = _git(origin, "ls-tree", "-r", "-z", "--name-only", "main").split("\0")
    for name in ('q"uote.json', "\u00e9.json", "with space.json"):
        assert f"runs/daily/{name}" in listed


@pytest.mark.parametrize("workflow", ["daily", "consistency", "brief"])
def test_each_workflow_notes_the_commit_it_checked_out(workflow: str) -> None:
    import yaml

    flow = yaml.safe_load(Path(f".github/workflows/{workflow}.yml").read_text())
    (job,) = flow["jobs"].values()
    steps = job["steps"]
    assert steps[0]["uses"].startswith("actions/checkout") and steps[0]["with"]["ref"] == "main"
    # read from git right after the checkout, before any step can change it
    assert steps[1]["run"] == 'echo "TENQ_CODE_SHA=$(git rev-parse HEAD)" >> "$GITHUB_ENV"'


def test_the_brief_never_moves_its_checkout() -> None:
    import yaml

    steps = yaml.safe_load(Path(".github/workflows/brief.yml").read_text())["jobs"]["brief"][
        "steps"
    ]
    # one checkout, first; no step moves it: records are saved from a copy of main
    checkouts = [i for i, s in enumerate(steps) if s.get("uses", "").startswith("actions/checkout")]
    assert checkouts == [0]
    movers = re.compile(r"\bgit\b[^\n]*\b(pull|fetch|reset|checkout|switch|merge|rebase)\b")
    moves = [i for i, s in enumerate(steps) if movers.search(s.get("run", ""))]
    assert moves == []
    assert steps[-1]["run"].strip().endswith("scripts/commit-daily.sh runs")
    (run,) = [s for s in steps if s.get("name") == "run"]
    assert run["env"]["TENQ_SAVE_CLAIMS"] == "1"  # each claim reaches main before its email


def test_the_brief_saves_all_of_runs(repos: tuple[Path, Path, Path]) -> None:
    origin, _, job = repos
    (job / "runs" / "CRWD").mkdir()
    (job / "runs" / "CRWD" / "0001.json").write_text("{}")
    (job / "runs" / "heartbeat").write_text("2026-11-07")
    assert _save(job, "runs").returncode == 0
    assert _on_main(origin, "runs/CRWD/0001.json") == "{}"
    assert _on_main(origin, "runs/heartbeat") == "2026-11-07"
    assert _save(job, "src").returncode == 2  # still only records folders


def test_a_file_saved_for_the_first_time_is_checked_against_the_checkout(
    repos: tuple[Path, Path, Path],
) -> None:
    """A later save compares a new file with the job's checkout, not with main
    as it was at the job's earlier save (which holds other writers' changes)."""
    origin, other, job = repos
    (other / "runs" / "B").mkdir()
    (other / "runs" / "B" / "x.json").write_text('{"sent": false}')
    _git(other, "add", ".")
    _git(other, "commit", "-q", "-m", "B")
    _git(other, "push", "-q", "origin", "HEAD:main")
    _git(job, "pull", "-q", "origin", "main")
    # a person marks B sent on main while the job runs
    (other / "runs" / "B" / "x.json").write_text('{"sent": true}')
    _git(other, "commit", "-q", "-am", "hand fix")
    _git(other, "push", "-q", "origin", "HEAD:main")
    (job / "runs" / "A").mkdir()
    (job / "runs" / "A" / "a.json").write_text("{}")
    assert _save(job, "runs").returncode == 0  # unrelated file: saved
    (job / "runs" / "B" / "x.json").write_text('{"sent": null, "status": "sending"}')
    got = _save(job, "runs")
    assert got.returncode == 1 and "main changed runs/B/x.json" in got.stdout
    assert _on_main(origin, "runs/B/x.json") == '{"sent": true}'


def test_the_job_can_save_its_own_file_again(repos: tuple[Path, Path, Path]) -> None:
    origin, _, job = repos
    (job / "runs" / "A").mkdir()
    for status in ("sending", "sent", "traced"):
        (job / "runs" / "A" / "a.json").write_text(status)
        assert _save(job, "runs").returncode == 0
        assert _on_main(origin, "runs/A/a.json") == status


def test_a_cut_off_write_in_a_ticker_folder_is_never_committed(
    repos: tuple[Path, Path, Path],
) -> None:
    origin, _, job = repos
    (job / "runs" / "A").mkdir()
    (job / "runs" / "A" / "a.json").write_text("{}")
    (job / "runs" / "A" / ".a.json.tmp").write_text("half")
    assert _save(job, "runs").returncode == 0
    assert "runs/A/.a.json.tmp" not in _git(origin, "ls-tree", "-r", "main")
    assert not (job / "runs" / "A" / ".a.json.tmp").exists()


def test_a_claim_counts_only_once_the_record_is_on_main(
    repos: tuple[Path, Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tenq.run import save_claim

    origin, _, job = repos
    record = job / "runs" / "CRWD" / "0001.json"
    record.parent.mkdir()
    record.write_text('{"email": {"sent": null, "status": "sending"}}')

    def refuse(*a: object, **kw: object) -> None:
        raise AssertionError("no save without TENQ_SAVE_CLAIMS")

    with monkeypatch.context() as m:
        m.setattr("tenq.run.subprocess.run", refuse)
        assert save_claim(record) is True  # off: nothing shared to claim on

    monkeypatch.setenv("TENQ_SAVE_CLAIMS", "1")
    assert save_claim(record) is True
    assert _on_main(origin, "runs/CRWD/0001.json") == record.read_text()

    # outside the saved folder: nothing reaches main, so no claim
    elsewhere = job / "records" / "0002.json"
    elsewhere.parent.mkdir()
    elsewhere.write_text("{}")
    assert save_claim(elsewhere) is False

    # a push that fails is no claim
    record.write_text('{"email": {"sent": null, "status": "sending", "try": 2}}')
    monkeypatch.setenv("TENQ_PUSH_URL", str(tmp_path / "nowhere.git"))
    assert save_claim(record) is False
