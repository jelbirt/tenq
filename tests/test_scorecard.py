from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from test_daily import Outbox, _ctx, _no_trace

from tenq.daily import main as daily_main
from tenq.daily_email import render
from tenq.prices import BENCHMARK, Bar, Prices
from tenq.scorecard import (
    calls_from,
    daily_records,
    last_session_of_week,
    lines,
    load_config,
    next_session,
    outcome,
    paired,
    planned_entry,
    planned_exit,
    rank_ic,
    read_ledger,
    scorecard,
    spearman,
    update,
    weekly,
)

CFG = load_config()
KNOWN_SCORES = {
    "score-v1": "2d1879cb5c1b613801681753bf688a870c5062bb4acac43f4f25223e85b90365",
}
MADE = datetime(2026, 11, 6, 11, 55, tzinfo=UTC)  # Friday 06:55 New York, before the open
ENTRY = date(2026, 11, 6)
EXIT = date(2026, 12, 4)  # the 20th session, counting Nov 6, skipping Thanksgiving


def test_the_score_settings_are_pinned_to_their_version() -> None:
    assert CFG.version in KNOWN_SCORES, "new score version: add it and its hash above"
    assert CFG.sha256 == KNOWN_SCORES[CFG.version], "score.yaml changed: bump the version"
    assert CFG.weights == {"Buy": 1.0, "Hold": 0.0, "Sell": -0.5, "Short": -1.0}
    assert CFG.sessions == 20


@pytest.mark.parametrize(
    ("bad", "says"),
    [
        ("version: v\n", "exactly"),
        (Path("config/score.yaml").read_text().replace("Sell: -0.5", "Sell: 0.5"), "weights"),
        (Path("config/score.yaml").read_text().replace("range: 0.90", "range: 1.5"), "range"),
    ],
)
def test_bad_score_settings_are_refused(tmp_path: Path, bad: str, says: str) -> None:
    path = tmp_path / "score.yaml"
    path.write_text(bad)
    with pytest.raises(ValueError, match=says):
        load_config(path)


@pytest.mark.parametrize(
    ("day", "last"),
    [
        (date(2026, 11, 6), True),  # a Friday
        (date(2026, 11, 5), False),
        (date(2026, 11, 25), False),  # Wednesday; Thanksgiving Thursday, Friday trades
        (date(2026, 11, 27), True),
        (date(2026, 4, 2), True),  # Thursday before Good Friday
        (date(2026, 12, 31), True),  # Thursday; New Year's Day is a holiday
    ],
)
def test_the_last_session_of_a_week(day: date, last: bool) -> None:
    assert last_session_of_week(day) is last


@pytest.mark.parametrize(
    ("made_at", "entry"),
    [
        (MADE, ENTRY),  # before Friday's open: Friday
        (datetime(2026, 11, 5, 22, 30, tzinfo=UTC), ENTRY),  # Thursday evening
        (datetime(2026, 11, 6, 14, 31, tzinfo=UTC), date(2026, 11, 9)),  # after the open
        (datetime(2026, 11, 6, 22, 30, tzinfo=UTC), date(2026, 11, 9)),  # Friday evening
        (datetime(2026, 11, 25, 22, 30, tzinfo=UTC), date(2026, 11, 27)),  # over Thanksgiving
    ],
)
def test_a_call_enters_at_the_first_open_after_it_was_made(made_at: datetime, entry: date) -> None:
    assert planned_entry(made_at) == entry


def test_the_exit_is_the_20th_session_counting_the_entry() -> None:
    assert planned_exit(ENTRY, 20) == EXIT
    assert planned_exit(ENTRY, 1) == ENTRY


def _sessions(n: int, start: date = ENTRY) -> list[date]:
    days = [start]
    while len(days) < n:
        days.append(next_session(days[-1]))
    return days


def _bars(days: list[date], open_: float, close: float, last_close: float) -> tuple[Bar, ...]:
    out = [Bar(d, close, None, open_) for d in days]
    out[-1] = replace(out[-1], close=last_close)
    return tuple(out)


def _prices(
    stock_days: int = 20, bench_days: int = 20, stock_open: float | None = 100.0
) -> dict[str, tuple[Bar, ...]]:
    stock = _bars(_sessions(stock_days), 100.0, 101.0, 110.0)
    stock = (replace(stock[0], open=stock_open), *stock[1:])
    return {
        "CRWD": stock,
        BENCHMARK: _bars(_sessions(bench_days), 200.0, 201.0, 204.0),
    }


def _stock(
    ticker: str = "CRWD",
    jev: str | None = "Buy",
    opus: str | None = "Short",
    score: float = 0.3,
    confidence: float = 0.6,
    momentum: float | None = 0.02,
    vol: float | None = 0.4,
) -> dict[str, Any]:
    numbers = []
    if momentum is not None:
        numbers.append({"id": f"px:{ticker}:excess_20d", "value": momentum})
    if vol is not None:
        numbers.append({"id": f"px:{ticker}:vol_20d", "value": vol})
    out: dict[str, Any] = {
        "ticker": ticker,
        "pack": {"numbers": numbers},
        "opus": {"call": opus, "confidence": confidence, "agent": {"model": "opus"}},
        "why": {"agent": {"model": "opus"}},
    }
    if jev:
        out["call"] = {"call": jev, "score": score, "confidence": 0.5}
    return out


def _record(
    day: str = "2026-11-05", made_at: datetime = MADE, *stocks: dict[str, Any], **kw: Any
) -> dict[str, Any]:
    return {
        "kind": "daily",
        "check_id": None,
        "complete": True,
        "day": day,
        "made_at": made_at.isoformat(),
        "versions": {"call_prompt": "call-v10"},
        "jev_cost": 0.03,
        "stocks": list(stocks) or [_stock()],
    } | kw


def _save(runs: Path, record: dict[str, Any], name: str | None = None) -> None:
    folder = runs / "daily"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / (name or f"{record['day']}.json")).write_text(json.dumps(record))


def test_a_finished_call_is_scored_from_the_open_to_the_20th_close() -> None:
    [made] = calls_from(_record())
    out = outcome(made, _prices(), CFG)
    assert out.status == "scored" and (out.entry_day, out.exit_day) == (ENTRY, EXIT)
    assert out.excess == pytest.approx((110 / 100 - 1) - (204 / 200 - 1))  # 10% less 2%


def test_a_call_short_of_20_sessions_is_open() -> None:
    [made] = calls_from(_record())
    out = outcome(made, _prices(19, 19), CFG)
    assert out.status == "open" and out.entry_day == ENTRY and out.excess is None


def test_a_stock_with_no_price_is_waited_for_then_left_unscored() -> None:
    [made] = calls_from(_record())
    assert outcome(made, _prices(15, 22), CFG).status == "open"  # QQQ only 2 sessions past
    out = outcome(made, _prices(15, 25), CFG)  # 5 sessions past the exit
    assert (out.status, out.reason) == ("unscored", "no price")


def test_a_missing_open_is_not_replaced_by_an_earlier_price() -> None:
    [made] = calls_from(_record())
    out = outcome(made, _prices(stock_open=None), CFG)
    assert (out.status, out.reason) == ("unscored", "no open price")


def test_only_full_finished_daily_records_are_read(tmp_path: Path) -> None:
    _save(tmp_path, _record("2026-11-05"))
    _save(tmp_path, _record("2026-11-06", check_id="ab12cd34"), "2026-11-06.check-ab12cd34.json")
    _save(tmp_path, _record("2026-11-09", complete=False))
    _save(tmp_path, _record("2026-11-10", kind="consistency"))
    (tmp_path / "daily" / "2026-11-11.json").write_text("{not json")
    assert [r["day"] for r in daily_records(tmp_path)] == ["2026-11-05"]


def test_a_stock_with_no_call_is_not_scored() -> None:
    record = _record("2026-11-05", MADE, _stock(jev=None, opus=None), _stock("NET", opus=None))
    [made] = calls_from(record)
    assert (made.ticker, made.jev, made.opus) == ("NET", "Buy", None)


def _fetcher(bars: dict[str, tuple[Bar, ...]], asked: list[Any] | None = None) -> Any:
    def fetch(tickers: list[str], period: str) -> Prices:
        if asked is not None:
            asked.append((tickers, period))
        return Prices(bars, ())

    return fetch


def test_a_finished_call_is_written_once_and_never_rescored(tmp_path: Path) -> None:
    _save(tmp_path, _record())
    asked: list[Any] = []
    done, opened, new = update(tmp_path, CFG, EXIT, _fetcher(_prices(), asked))
    assert new == 1 and opened == [] and asked == [(["CRWD"], "6mo")]
    [row] = done
    assert row["excess"] == pytest.approx(0.08) and row["score_version"] == "score-v1"
    worse = {**_prices(), "CRWD": _bars(_sessions(20), 100.0, 101.0, 50.0)}
    done, _, new = update(tmp_path, CFG, EXIT, _fetcher(worse, asked))
    assert new == 0 and done[0]["excess"] == pytest.approx(0.08)
    assert len(asked) == 1  # nothing left to fetch


def test_a_call_not_due_is_open_without_a_price_fetch(tmp_path: Path) -> None:
    _save(tmp_path, _record())
    asked: list[Any] = []
    done, opened, new = update(tmp_path, CFG, EXIT - timedelta(days=1), _fetcher({}, asked))
    assert (done, new, asked) == ([], 0, [])
    assert opened[0]["exit_day"] == "2026-12-04"


def test_without_save_the_ledger_is_untouched(tmp_path: Path) -> None:
    _save(tmp_path, _record())
    update(tmp_path, CFG, EXIT, _fetcher(_prices()), save=False)
    assert not (tmp_path / "daily" / "scores").exists()


def test_a_ledger_from_other_settings_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "daily" / "scores" / "score-v1.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"version": "score-v1", "sha256": "other", "rows": {}}))
    with pytest.raises(ValueError, match="other score settings"):
        read_ledger(path, CFG)


def _rows(*items: tuple[str, str | None, str | None, float]) -> list[dict[str, Any]]:
    return [
        {"day": d, "ticker": f"T{i}", "jev": j, "opus": o, "excess": x}
        for i, (d, j, o, x) in enumerate(items)
    ]


def _weeks(n: int, jev: str, opus: str, excess: float) -> list[dict[str, Any]]:
    start = date(2026, 11, 2)
    return _rows(*[((start + timedelta(weeks=w)).isoformat(), jev, opus, excess) for w in range(n)])


def test_the_verdict_waits_for_enough_weeks_whatever_the_numbers() -> None:
    p = paired(_weeks(12, "Buy", "Hold", 0.05), CFG)
    assert p["verdict"] == "too early to tell (12 of 13 weeks scored)"
    assert p["mean"] == pytest.approx(0.05)


def test_a_method_is_ahead_only_when_the_whole_range_is_on_its_side() -> None:
    assert paired(_weeks(13, "Buy", "Hold", 0.05), CFG)["verdict"] == "Jev ahead"
    assert paired(_weeks(13, "Hold", "Buy", 0.05), CFG)["verdict"] == "Opus ahead"
    mixed = _weeks(7, "Buy", "Hold", 0.05) + _weeks(7, "Buy", "Hold", -0.05)
    mixed = [
        r | {"day": (date(2026, 11, 2) + timedelta(weeks=i)).isoformat()}
        for i, r in enumerate(mixed)
    ]
    p = paired(mixed, CFG)
    assert p["verdict"] == "no clear difference yet" and p["low"] < 0 < p["high"]
    assert paired(mixed, CFG) == p  # the same seed, the same range


def test_only_calls_both_made_are_paired() -> None:
    p = paired(_rows(("2026-11-05", "Buy", None, 0.1), ("2026-11-05", "Buy", "Sell", 0.1)), CFG)
    assert p["pairs"] == 1 and p["mean"] == pytest.approx(0.1 - (-0.05))


def test_rank_correlation_handles_ties() -> None:
    assert spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman([1, 1, 2], [3, 2, 1]) == pytest.approx(-0.8660254)
    assert spearman([1, 1, 1], [1, 2, 3]) is None


def test_rank_ic_needs_five_stocks_a_day() -> None:
    rows = [
        {"day": "2026-11-05", "jev_score": s, "excess": s / 10} for s in (0.1, 0.2, 0.3, 0.4, 0.5)
    ]
    assert rank_ic(rows, "jev", CFG) == pytest.approx(1.0)
    assert rank_ic(rows[:4], "jev", CFG) is None


def _card(tmp_path: Path, *records: dict[str, Any], today: date = EXIT) -> dict[str, Any]:
    for r in records:
        _save(tmp_path, r)
    return scorecard(tmp_path, today, CFG, _fetcher(_prices()), save=False)


def test_the_scorecard_scores_both_methods_and_the_baselines(tmp_path: Path) -> None:
    card = _card(tmp_path, _record())
    assert card["scored"] == 1 and card["open"] == 0
    jev, opus = card["methods"]["jev"], card["methods"]["opus"]
    assert jev["mean"] == pytest.approx(0.08) and opus["mean"] == pytest.approx(-0.08)
    assert jev["hit_rate"] == 1.0 and opus["hit_rate"] == 0.0
    assert jev["by_sell_weight"] == {
        "0.25": pytest.approx(0.08),
        "0.5": pytest.approx(0.08),
        "1": pytest.approx(0.08),
    }
    assert jev["mean_scaled"] == pytest.approx(0.08 / (0.4 * (20 / 252) ** 0.5))
    assert card["baselines"] == {
        "always_hold": 0.0,
        "always_buy": pytest.approx(0.08),
        "last_months_winners": pytest.approx(0.08),
        "beat_qqq": 1.0,
    }
    assert card["disagreements"] == {
        "count": 1,
        "jev": pytest.approx(0.08),
        "opus": pytest.approx(-0.08),
    }
    assert card["paired"]["mean"] == pytest.approx(0.16)


def test_the_words_put_the_verdict_first_and_name_the_open_calls(tmp_path: Path) -> None:
    card = _card(tmp_path, _record(), today=EXIT - timedelta(days=1))
    text = lines(card)
    assert text[0] == "Verdict, Jev against Opus: too early to tell (0 of 13 weeks scored)."
    assert "1 call(s) still open; the next are scored after the close on 2026-12-04." in text
    assert "No call has finished its 20 sessions yet." in text
    assert text[-1].startswith("This week's cost: Jev $0.00 over 0 day(s)")


def test_weekly_runs_only_on_the_weeks_last_session(tmp_path: Path) -> None:
    def refuse(tickers: list[str], period: str) -> Prices:
        raise AssertionError("no fetch")

    assert weekly(tmp_path, date(2026, 11, 5), refuse) is None
    card = weekly(tmp_path, date(2026, 11, 6), refuse)
    assert card is not None and card["lines"][0].startswith("Verdict")


def test_a_failing_scorecard_is_an_error_line_not_a_crash(tmp_path: Path) -> None:
    _save(tmp_path, _record("2026-10-01", datetime(2026, 10, 1, 22, 30, tzinfo=UTC)))

    def broken(tickers: list[str], period: str) -> Prices:
        raise OSError("feed down")

    assert weekly(tmp_path, date(2026, 11, 6), broken) == {
        "error": "the scorecard could not be made (OSError)"
    }


FRIDAY = datetime(2026, 11, 6, 22, 30, tzinfo=UTC)


def _friday(tmp_path: Path, score_prices: Any, outbox: Outbox) -> dict[str, Any]:
    ctx = _ctx()
    ctx.now = FRIDAY
    code = daily_main(
        ["--runs", str(tmp_path), "--stage", "all"],
        ctx=ctx,
        prices=lambda: Prices({BENCHMARK: (Bar(date(2026, 11, 6), 600.0, None),)}, ()),
        score_prices=score_prices,
        mailer=outbox,
        tracer=_no_trace,
    )
    assert code == 0
    out: dict[str, Any] = json.loads((tmp_path / "daily" / "2026-11-06.json").read_text())
    return out


def test_fridays_email_leads_with_the_scorecard(tmp_path: Path) -> None:
    outbox = Outbox()
    record = _friday(tmp_path, _fetcher({}), outbox)
    assert record["scorecard"]["lines"][0].startswith("Verdict, Jev against Opus: too early")
    [email] = outbox.sent
    assert email.subject.startswith("Weekly scorecard + tenq daily, session 2026-11-06")
    assert email.text.startswith("WEEKLY SCORECARD\nVerdict, Jev against Opus")
    assert email.html.index("Weekly scorecard") < email.html.index("tenq daily calls")


def test_a_broken_scorecard_still_sends_the_email_and_says_so(tmp_path: Path) -> None:
    _save(tmp_path, _record("2026-10-01", datetime(2026, 10, 1, 22, 30, tzinfo=UTC)))

    def broken(tickers: list[str], period: str) -> Prices:
        raise OSError("feed down")

    outbox = Outbox()
    _friday(tmp_path, broken, outbox)
    [email] = outbox.sent
    assert email.subject.startswith("tenq daily, session 2026-11-06")
    assert email.text.startswith(
        "WEEKLY SCORECARD\nWeekly scorecard: the scorecard could not be made (OSError)."
    )


def test_a_midweek_email_has_no_scorecard(tmp_path: Path) -> None:
    record = _record()
    assert "scorecard" not in record
    email = render(
        record
        | {
            "run_id": "r",
            "versions": {"questions": "q", "rules": "r", "call_prompt": "c", "why_prompt": "w"},
            "stocks": [],
        }
    )
    assert not email.subject.startswith("Weekly") and "WEEKLY" not in email.text
