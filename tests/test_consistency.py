from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from test_daily import Claude, Jev
from test_daily import _ctx as _daily_ctx

from tenq.consistency import in_daily_window, lines, main, summarize
from tenq.daily import Context
from tenq.jev import Result
from tenq.llm import AgentRun

SATURDAY = datetime(2026, 11, 7, 15, 0, tzinfo=UTC)  # outside the daily run's window


def _ctx(*args: Any, **kw: Any) -> Context:
    ctx = _daily_ctx(*args, **kw)
    ctx.now = SATURDAY
    return ctx


def _run(tmp_path: Path, *argv: str, **kw: Any) -> tuple[int, dict[str, Any]]:
    code = main(["--runs", str(tmp_path), *argv], ctx=_ctx(**kw))
    files = list((tmp_path / "consistency").glob("*.json"))
    record = json.loads(files[0].read_text()) if files else {}
    return code, record


def test_each_method_is_asked_repeats_times_on_one_pack(tmp_path: Path) -> None:
    gathered: list[str] = []
    jev, claude = Jev(), Claude()
    ctx = _ctx(jev, claude)
    real = ctx.gather

    def gather(ticker: str, now: Any) -> Any:
        gathered.append(ticker)
        return real(ticker, now)

    ctx.gather = gather
    assert main(["--runs", str(tmp_path), "--ticker", "CRWD", "--ticker", "NET"], ctx=ctx) == 0
    assert gathered == ["CRWD", "NET"]  # one pack per stock, shared by every run
    assert len(jev.asked) == 6 and len(set(jev.asked)) == 6  # a fresh session per run
    assert len(claude.calls) == 6
    assert not any("How the rules reached it" in m for m in claude.calls)  # no why asked
    (path,) = (tmp_path / "consistency").glob("*.json")
    assert path.name.startswith("2026-11-07-")  # the date it ran
    record = json.loads(path.read_text())
    assert record["kind"] == "consistency" and record["repeats"] == 3
    assert record["complete"] is True and record["day"] == "2026-11-04"
    assert record["versions"]["call_prompt"].startswith("call-v")
    assert record["versions"]["rules_sha256"]
    assert all(record["run_id"][:8] in a for a in jev.asked)  # no clash with another run
    assert record["summary"]["jev"] == {
        "asked": 2,
        "stocks": 2,
        "same_call": 2,
        "changed": [],
        "measure": "score",
        "median_move": 0.0,
        "largest_move": 0.0,
    }
    assert record["summary"]["opus"]["same_call"] == 2


def test_a_changed_call_is_counted_and_named() -> None:
    stocks = [
        {
            "ticker": "A",
            "opus": [{"call": "Buy", "confidence": 0.6}, {"call": "Hold", "confidence": 0.4}],
        },
        {
            "ticker": "B",
            "opus": [{"call": "Hold", "confidence": 0.4}, {"call": "Hold", "confidence": 0.5}],
        },
    ]
    s = summarize(stocks, "opus", "confidence")
    assert s["stocks"] == 2 and s["same_call"] == 1
    assert s["changed"] == [{"ticker": "A", "calls": ["Buy", "Hold"]}]
    assert round(s["median_move"], 6) == 0.15
    assert round(s["largest_move"], 6) == 0.2
    text = lines({"jev": summarize([], "jev", "score"), "opus": s}, 2)
    assert text[0] == "Jev: no stock got a call on every run"
    assert text[1].startswith("Opus: same call on 1 of 2 stocks with all 2 runs answered; confid")
    assert text[2] == "  A changed: Buy / Hold"


def test_a_run_that_failed_is_left_out_not_read_as_a_change() -> None:
    stocks = [{"ticker": "A", "jev": [{"call": "Buy", "score": 0.3}, {"call": None, "error": "x"}]}]
    s = summarize(stocks, "jev", "score")
    assert s["stocks"] == 0 and s["changed"] == [] and s["asked"] == 1
    assert lines({"jev": s, "opus": s}, 2)[0] == (
        "Jev: no stock got a call on every run (1 more left out: a run failed)"
    )


def test_no_opus_asks_jev_only(tmp_path: Path) -> None:
    claude = Claude()
    code, record = _run(tmp_path, "--ticker", "CRWD", "--no-opus", claude=claude)
    assert code == 0 and claude.calls == []
    assert record["opus_on"] is False
    assert record["stocks"][0]["opus"][0]["error"] == "Opus was off for this run"
    assert record["summary"]["opus"]["stocks"] == 0


def test_claude_at_its_limit_stops_opus_for_the_rest(tmp_path: Path) -> None:
    claude = Claude(limit_after=1)
    code, record = _run(tmp_path, "--ticker", "CRWD", "--ticker", "NET", claude=claude)
    assert code == 0
    assert len(claude.calls) == 2  # the second hit the limit; nothing asked after
    crwd, net = record["stocks"]
    assert [r["call"] for r in crwd["opus"]] == ["Hold", None, None]
    assert net["opus"][0]["error"] == "not asked: Claude's usage limit was hit"
    assert record["summary"]["jev"]["stocks"] == 2  # Jev still ran


def test_jev_down_is_not_asked_again(tmp_path: Path) -> None:
    jev = Jev(down_after=1)
    code, record = _run(tmp_path, "--ticker", "CRWD", "--no-opus", jev=jev)
    assert code == 0 and len(jev.asked) == 2
    assert record["stocks"][0]["jev"][2]["error"] == "not asked: Jev was down earlier in this run"


def test_a_failure_in_code_is_recorded_on_the_stock(tmp_path: Path) -> None:
    class Broken(Claude):
        def __call__(self, message: str, **kw: Any) -> AgentRun:
            raise RuntimeError("secret detail")

    code, record = _run(tmp_path, "--ticker", "CRWD", claude=Broken())
    assert code == 0
    assert record["stocks"][0]["opus"] == [{"call": None, "error": "failed in code (RuntimeError)"}]
    assert len(record["stocks"][0]["jev"]) == 3  # Jev's runs are kept
    assert "secret detail" not in json.dumps(record)


def test_no_pack_for_any_stock_exits_1(tmp_path: Path) -> None:
    ctx = _ctx()
    ctx.gather = lambda ticker, now: None
    assert main(["--runs", str(tmp_path), "--ticker", "CRWD"], ctx=ctx) == 1


def test_bad_arguments_exit_2(tmp_path: Path) -> None:
    assert main(["--runs", str(tmp_path), "--repeats", "1"], ctx=_ctx()) == 2
    assert main(["--runs", str(tmp_path), "--repeats", "9"], ctx=_ctx()) == 2
    assert main(["--runs", str(tmp_path), "--ticker", "ZZZZ"], ctx=_ctx()) == 2
    assert not (tmp_path / "consistency").exists()


def test_a_run_cut_off_keeps_the_stocks_it_finished(tmp_path: Path) -> None:
    ctx = _ctx()
    real = ctx.gather

    def gather(ticker: str, now: Any) -> Any:
        if ticker == "NET":
            raise KeyboardInterrupt  # the job's time limit, say: not caught
        return real(ticker, now)

    ctx.gather = gather
    with pytest.raises(KeyboardInterrupt):
        main(["--runs", str(tmp_path), "--ticker", "CRWD", "--ticker", "NET"], ctx=ctx)
    (path,) = (tmp_path / "consistency").glob("*.json")
    record = json.loads(path.read_text())
    assert record["complete"] is False
    assert [s["ticker"] for s in record["stocks"]] == ["CRWD"]
    assert record["summary"]["jev"]["stocks"] == 1


def test_every_run_failing_exits_1(tmp_path: Path) -> None:
    def failing(state: str, questions: Any, **kw: Any) -> Result:
        return Result("q", "m", error="no key")

    ctx = _ctx()
    ctx.ask = failing
    assert main(["--runs", str(tmp_path), "--ticker", "CRWD", "--no-opus"], ctx=ctx) == 1


def test_a_failure_part_way_keeps_the_runs_done() -> None:
    from tenq.consistency import measure
    from tenq.daily import State

    calls = 0

    class Flaky(Jev):
        def __call__(self, state: str, questions: Any, **kw: Any) -> Result:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ValueError("boom")
            return super().__call__(state, questions, **kw)

    ctx = _ctx(Flaky())
    ctx.claude = None
    out = measure("CRWD", "CrowdStrike", ctx, State(deadline=1e18), 3, "abcdef12")
    assert [r["call"] for r in out["jev"]] == ["Buy", None]
    assert out["jev"][1]["error"] == "failed in code (ValueError)"


@pytest.mark.parametrize(
    ("when", "inside"),
    [
        (datetime(2026, 11, 4, 22, 30, tzinfo=UTC), True),  # Wednesday, the daily run
        (datetime(2026, 11, 4, 19, 59, tzinfo=UTC), False),
        (datetime(2026, 11, 5, 1, 59, tzinfo=UTC), True),  # Thursday, just after midnight
        (datetime(2026, 11, 5, 2, 0, tzinfo=UTC), False),
        (datetime(2026, 11, 7, 21, 0, tzinfo=UTC), False),  # Saturday evening
        (datetime(2026, 11, 7, 1, 0, tzinfo=UTC), True),  # Friday night's run, past midnight
        (datetime(2026, 11, 9, 1, 0, tzinfo=UTC), False),  # Sunday night: no run before
    ],
)
def test_the_daily_window(when: datetime, inside: bool) -> None:
    assert in_daily_window(when) is inside


def test_opus_is_refused_in_the_daily_window_and_above_the_cap(tmp_path: Path) -> None:
    ctx = _ctx()
    ctx.now = datetime(2026, 11, 4, 22, 30, tzinfo=UTC)
    assert main(["--runs", str(tmp_path), "--ticker", "CRWD"], ctx=ctx) == 2
    ctx = _ctx()
    ctx.now = datetime(2026, 11, 4, 22, 30, tzinfo=UTC)
    assert main(["--runs", str(tmp_path), "--ticker", "CRWD", "--no-opus"], ctx=ctx) == 0
    assert main(["--runs", str(tmp_path), "--repeats", "4"], ctx=_ctx()) == 2  # 4 x 20 > 60


def test_the_opus_budget_skips_opus_but_not_jev(tmp_path: Path) -> None:
    claude = Claude()
    ctx = _ctx(claude=claude)
    ticks = iter(range(0, 10**9, 3000))
    ctx.clock = lambda: float(next(ticks))  # 50 minutes a tick
    code, record = _run_ctx(tmp_path, ctx, "--ticker", "CRWD", "--ticker", "NET")
    assert code == 0
    net = record["stocks"][1]
    assert net["opus"][-1]["error"] == "not asked: the run's time budget for Opus ran out"
    assert all(r["call"] for r in net["jev"])


def _run_ctx(tmp_path: Path, ctx: Context, *argv: str) -> tuple[int, dict[str, Any]]:
    code = main(["--runs", str(tmp_path), *argv], ctx=ctx)
    (path,) = (tmp_path / "consistency").glob("*.json")
    return code, json.loads(path.read_text())


def test_the_commit_script_takes_only_a_records_folder() -> None:
    got = subprocess.run(
        ["bash", "scripts/commit-daily.sh", "src"], capture_output=True, text=True, check=False
    )
    assert got.returncode == 2 and "not a records folder" in got.stdout
