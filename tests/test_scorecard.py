from __future__ import annotations

import json
import random
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
    _lead_lines,
    _standard_error,
    calls_from,
    daily_records,
    day_totals,
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
    t95,
    update,
    weekly,
)

CFG = load_config()
KNOWN_SCORES = {
    "score-v1": "6f1a5031248f812ac0841e2e329e3cd20fd66842f7898d3d061d83053090009a",
    # v2: the luck range from daily moves, and a settled bar for weekly reading
    "score-v2": "bfd22fe3774cf9e1ea163b244f31a87ab12274ca5fb95810b5f3f8859e3bfd24",
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
        (
            Path("config/score.yaml").read_text().replace("settled_bar: 2.4", "settled_bar: 0.5"),
            "settled_bar",
        ),
        (
            Path("config/score.yaml").read_text().replace("carry_days: 2", "carry_days: -1"),
            "carry_days",
        ),
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
        "versions": {"call_prompt": "call-v11"},
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
    assert outcome(made, _prices(stock_open=None), CFG).status == "open"  # may come in later
    out = outcome(made, _prices(20, 25, stock_open=None), CFG)
    assert (out.status, out.reason) == ("unscored", "no open price")


def test_a_stock_with_nothing_downloaded_waits_far_longer_than_a_gap() -> None:
    """One failed download looks the same as a delisting, so it is not final."""
    [made] = calls_from(_record())
    gone = _prices(20, 20 + CFG.no_download_after - 1)
    del gone["CRWD"]
    assert outcome(made, gone, CFG).status == "open"
    gone = _prices(20, 20 + CFG.no_download_after)
    del gone["CRWD"]
    out = outcome(made, gone, CFG)
    assert (out.status, out.reason) == ("unscored", "no price downloaded")


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
    assert row["excess"] == pytest.approx(0.08) and row["score_version"] == CFG.version
    assert sum(row["moves"]) == pytest.approx(row["excess"]) and len(row["moves"]) == 20
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


def test_a_lead_waits_for_enough_weeks_whatever_the_numbers() -> None:
    p = paired(_weeks(12, "Buy", "Hold", 0.05), CFG)
    assert p["leader"] == "Jev" and p["settled"] is None and p["weeks"] == 12
    assert p["mean"] == pytest.approx(0.05)


def _held(day: date) -> list[str]:
    """The 20 sessions a call made on `day` is held, from the next one on."""
    return [d.isoformat() for d in _sessions(20, next_session(day))]


def _skilled(weeks: int, gap: float, seed: int = 7) -> list[dict[str, Any]]:
    """Twenty stocks, one call each a week, where Jev's Buy beats Opus's Hold
    by about `gap` a call, stored with daily moves as the ledger stores them;
    the stocks share a market move each day, half up and half down."""
    rng = random.Random(seed)
    days = _sessions(weeks * 5 + 21, date(2026, 1, 5))
    market = {d: rng.gauss(0, 0.01) for d in days}
    rows = []
    for w in range(weeks):
        made = days[w * 5]
        held = _held(made)
        for t in range(20):
            lean = 1 if t % 2 else -1
            moves = [gap / 20 + lean * market[date.fromisoformat(d)] + rng.gauss(0, 0.01)
                     for d in held]  # fmt: skip
            rows.append({"day": made.isoformat(), "ticker": f"T{t}", "jev": "Buy",
                         "opus": "Hold", "excess": sum(moves), "moves": moves,
                         "move_days": held, "entry_day": held[0],
                         "exit_day": held[-1]})  # fmt: skip
    return rows


def test_the_leader_shows_from_the_first_pair_and_settles_only_when_clear() -> None:
    early = paired(_skilled(3, 0.03), CFG)
    assert early["leader"] == "Jev" and early["settled"] is None and early["weeks"] == 3
    clear = paired(_skilled(13, 0.03), CFG)
    assert clear["settled"] == "Jev"
    assert clear["luck"] == pytest.approx(t95(clear["days"] - 1) * clear["se"])
    assert clear["bar"] == pytest.approx(CFG.settled_bar * clear["se"])
    flipped = paired([r | {"jev": "Hold", "opus": "Buy"} for r in _skilled(13, 0.03)], CFG)
    assert (flipped["leader"], flipped["settled"]) == ("Opus", "Opus")
    assert flipped["mean"] == pytest.approx(-clear["mean"])
    faint = paired(_skilled(13, 0.0005), CFG)
    assert faint["leader"] == "Jev" and faint["settled"] is None
    assert faint["luck"] > abs(faint["mean"])


def test_a_lead_past_the_luck_range_but_short_of_the_bar_is_not_settled() -> None:
    """The card is read every week, so a lead outside the one-look range is
    still not settled until it clears the stricter bar, and the card says so."""
    between = None
    for step in range(1, 400):
        p = paired(_skilled(13, step / 10000), CFG)
        if p["luck"] < abs(p["mean"]) < p["bar"]:
            between = step / 10000
            break
    assert between is not None
    p = paired(_skilled(13, between), CFG)
    assert p["settled"] is None
    assert paired(_skilled(13, between), replace(CFG, settled_bar=1.0))["settled"] == "Jev"
    text = _lead_lines(p, 13)
    assert text[1] == (
        f"Not settled: 9 times in 10, luck alone gives a gap between -{p['luck'] * 100:.2f}% "
        f"and +{p['luck'] * 100:.2f}% at this point."
    )
    assert text[2] == (
        "The gap is outside that range, but since the card is read every week a lead "
        f"settles only at {p['bar'] * 100:.2f}%."
    )


def test_an_early_lead_that_clears_the_bar_is_told_it_waits_for_the_weeks() -> None:
    p = paired(_skilled(5, 0.03), CFG)
    assert p["settled"] is None and abs(p["mean"]) >= p["bar"]
    text = _lead_lines(p, 13)
    assert text[1].startswith("Not settled (too early: 5 of 13 weeks scored): 9 times in 10,")
    assert text[2] == (
        f"The gap already clears the settle bar of {p['bar'] * 100:.2f}%; a lead can settle "
        "only once 13 weeks are scored."
    )


def test_a_gap_that_rounds_to_zero_has_no_leader() -> None:
    rows = _skilled(13, 0.0)
    base = paired(rows, CFG)["mean"]
    tiny = [r | {"excess": r["excess"] - base + 0.00004} for r in rows]
    p = paired([r | {"moves": None} for r in tiny], CFG)
    assert 0 < p["mean"] < 0.00005 and p["leader"] is None and p["differ"] == len(rows)
    text = _lead_lines(p, 13)
    assert text[0].startswith("Ahead so far: neither, level (under 0.01% apart) over ")
    assert text[1].startswith("Not settled: 9 times in 10, luck alone gives a gap between")


def test_the_standard_error_matches_a_hand_worked_case() -> None:
    """Day totals 1, 3, 2, 6 over 4 pairs: deviations -2, 0, -1, 3 give a
    spread of 14 and a one-day carry-over of -3 at weight 1/2, so 11; times
    4/3 for a short sample, square root, over 4 pairs. Two days of carry-over
    add the two-day term, 2, at weight 1/3 and the one-day term at 2/3."""
    totals = [1.0, 3.0, 2.0, 6.0]
    assert _standard_error(totals, 4, 1) == pytest.approx((11 * 4 / 3) ** 0.5 / 4)
    assert _standard_error(totals, 4, 0) == pytest.approx((14 * 4 / 3) ** 0.5 / 4)
    assert _standard_error(totals, 4, 2) == pytest.approx(
        ((14 + 2 * (2 / 3) * -3 + 2 * (1 / 3) * 2) * 4 / 3) ** 0.5 / 4
    )
    assert _standard_error([1.0], 4, 1) is None


def test_calls_that_all_agree_claim_no_range_and_never_settle() -> None:
    rows = [r | {"opus": "Buy"} for r in _skilled(13, 0.03)]
    p = paired(rows, CFG)
    assert (p["leader"], p["luck"], p["settled"], p["differ"]) == (None, None, None, 0)
    assert _lead_lines(p, 13)[:2] == [
        f"Ahead so far: neither, level over {p['pairs']} pair(s) in 13 week(s).",
        "Not settled: the two have not disagreed on a finished call yet.",
    ]


def test_daily_moves_land_on_the_sessions_they_were_measured_on() -> None:
    """Two calls made on the same day land on their 20 sessions, skipping
    Thanksgiving and ending on the exit day; a row whose moves do not match
    its days or its excess falls back to its exit day."""
    held = _held(date(2026, 11, 5))
    rows = [
        {"day": "2026-11-05", "ticker": t, "jev": "Buy", "opus": "Hold", "excess": 0.02,
         "moves": [0.001] * 20, "move_days": held, "entry_day": held[0],
         "exit_day": held[-1]}
        for t in ("A", "B")
    ]  # fmt: skip
    totals = day_totals([(1.0, r) for r in rows])
    assert [d.isoformat() for d in totals] == held
    assert held[-1] == "2026-12-04" and "2026-11-26" not in held
    p = paired(rows, CFG)
    assert (p["pairs"], p["days"]) == (2, 20) and p["mean"] == pytest.approx(0.02)
    for bad in ({"moves": [0.5] * 20}, {"move_days": held[:19]}, {"move_days": None}):
        assert list(day_totals([(1.0, r | bad) for r in rows])) == [date(2026, 12, 4)]


def _climbing(days: list[date], start: float, step: float) -> list[Bar]:
    return [Bar(d, start + step * (i + 1), None, start) for i, d in enumerate(days)]


def test_moves_add_up_to_the_excess_through_a_missing_stock_bar() -> None:
    """A day the stock has no bar moves only by QQQ; the next bar carries the
    stock's change, and the moves still add up to the excess exactly."""
    [made] = calls_from(_record())
    days = _sessions(20)
    stock = _climbing(days, 100.0, 1.0)
    del stock[5]
    bench = _climbing(days, 200.0, 0.5)
    out = outcome(made, {"CRWD": tuple(stock), BENCHMARK: tuple(bench)}, CFG)
    assert out.status == "scored" and out.moves is not None
    assert out.moves[5] == pytest.approx(-0.5 / 200)
    assert out.moves[6] == pytest.approx(2.0 / 100 - 0.5 / 200)
    assert sum(out.moves) == pytest.approx(out.excess, abs=1e-12)
    assert out.move_days == tuple(days)


def test_moves_follow_qqqs_bars_when_one_is_missing() -> None:
    """If QQQ's download skips a session the call is held one session longer,
    and each move stays on the bar it was measured on."""
    [made] = calls_from(_record())
    days = _sessions(21)
    bench = _climbing(days, 200.0, 0.5)
    del bench[5]
    stock = _climbing(days, 100.0, 1.0)
    out = outcome(made, {"CRWD": tuple(stock), BENCHMARK: tuple(bench)}, CFG)
    assert out.status == "scored" and out.move_days is not None
    assert out.move_days == tuple(b.day for b in bench[:20])
    assert out.move_days[-1] == out.exit_day == days[20]
    assert sum(out.moves or ()) == pytest.approx(out.excess, abs=1e-12)


def test_t95_matches_the_table_and_meets_the_normal() -> None:
    assert t95(1) == pytest.approx(6.314)
    assert t95(12) == pytest.approx(1.782)
    assert 1.782 > t95(13) > 1.753
    assert t95(10**6) == pytest.approx(1.645, abs=1e-3)
    with pytest.raises(ValueError):
        t95(0)


def _null_trial(seed: int, weeks: int = 13, stocks: int = 20) -> list[dict[str, Any]]:
    """Two methods with no skill whose calls stick for weeks, as real calls do,
    on stocks whose 20-session excess returns overlap from day to day and that
    share a market move each day, stored with their daily moves as the ledger
    stores them."""
    rng = random.Random(seed)
    days = _sessions(weeks * 5 + 20, date(2026, 1, 5))
    names = list(CFG.weights)
    rows = []
    market = [rng.gauss(0, 0.015) for _ in days]
    for s in range(stocks):
        lean = 1 if s % 2 else -1
        daily = [rng.gauss(0, 0.02) + lean * m for m in market]
        jev, opus = rng.choice(names), rng.choice(names)
        for i in range(weeks * 5):
            if rng.random() < 0.05:
                jev = rng.choice(names)
            if rng.random() < 0.05:
                opus = rng.choice(names)
            moves = daily[i + 1 : i + 21]
            rows.append({"day": days[i].isoformat(), "ticker": f"T{s}", "jev": jev,
                         "opus": opus, "excess": sum(moves), "moves": moves,
                         "move_days": [d.isoformat() for d in days[i + 1 : i + 21]],
                         "entry_day": days[i + 1].isoformat(),
                         "exit_day": days[i + 20].isoformat()})  # fmt: skip
    return rows


def test_sticky_calls_with_no_skill_rarely_settle() -> None:
    """Resampling single weeks named a winner in about 40% of such trials;
    the settled bar is meant to do so about 10% of the time or less."""
    wins = sum(bool(paired(_null_trial(seed), CFG)["settled"]) for seed in range(40))
    assert wins <= 3


def test_the_luck_range_has_the_right_size_when_neither_method_has_skill() -> None:
    """Over many no-skill trials the gap divided by its standard error should
    spread like a standard normal: about 1. A formula that over- or under-states
    luck by 15% or more fails here."""
    ratios = [p["mean"] / p["se"] for p in (paired(_null_trial(s), CFG) for s in range(100))]
    spread = (sum(r * r for r in ratios) / len(ratios)) ** 0.5
    assert 0.85 <= spread <= 1.15


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


def _opus_rows(leans: list[float | None]) -> list[dict[str, Any]]:
    """Five Opus Holds at 40% (no spread by call times confidence), with leans
    that order the stocks the way their excess returns do."""
    return [
        {"day": "2026-11-05", "opus": "Hold", "opus_confidence": 0.4, "opus_lean": v, "excess": i}
        for i, v in enumerate(leans)
    ]


def test_opus_is_ranked_by_its_lean_when_every_call_that_day_has_one() -> None:
    assert rank_ic(_opus_rows([-0.4, -0.2, 0.0, 0.1, 0.3]), "opus", CFG) == pytest.approx(1.0)


def _mixed_day(day: str, missing: bool) -> list[dict[str, Any]]:
    """Six Opus Buys whose confidence orders the stocks against their excess
    returns while their leans order them with it."""
    rows = [
        {"day": day, "opus": "Buy", "opus_confidence": 0.9 - i / 10, "opus_lean": 0.1 + i / 10}
        | {"excess": i}
        for i in range(6)
    ]
    if missing:
        rows[2]["opus_lean"] = None
    return rows


def test_one_missing_lean_ranks_the_whole_day_by_call_and_confidence() -> None:
    assert rank_ic(_mixed_day("2026-11-05", False), "opus", CFG) == pytest.approx(1.0)
    # one lean missing: all six are ranked by weight times confidence, none dropped
    assert rank_ic(_mixed_day("2026-11-05", True), "opus", CFG) == pytest.approx(-1.0)


def test_each_day_picks_its_own_ranking_and_the_days_are_averaged() -> None:
    rows = _mixed_day("2026-11-05", False) + _mixed_day("2026-11-06", True)
    assert rank_ic(rows, "opus", CFG) == pytest.approx(0.0)


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


def test_the_words_put_the_leader_first_and_name_the_open_calls(tmp_path: Path) -> None:
    card = _card(tmp_path, _record(), today=EXIT - timedelta(days=1))
    text = lines(card)
    assert text[0] == "Ahead so far: no recommendation both made has finished yet."
    assert (
        "1 call(s) still open; the next finish on 2026-12-04. Finished calls are scored in "
        "the next weekly card."
    ) in text
    assert "No call has finished its 20 sessions yet." in text
    assert text[-1].startswith("This week's cost: Jev $0.00 over 0 day(s)")


def test_unscored_calls_are_one_line_per_stock(tmp_path: Path) -> None:
    card = {
        "paired": {"pairs": 0},
        "open": 0, "scored": 0,
        "unscored": [
            {"day": d, "ticker": "XYZ", "reason": "no price downloaded"}
            for d in ("2026-11-05", "2026-11-06", "2026-11-09")
        ] + [{"day": "2026-11-05", "ticker": "ABC", "reason": "no open price"}],
        "this_week": {"jev_usd": 0.0, "days": 0, "opus_runs": 0, "opus_calls": 0, "opus_whys": 0},
    }  # fmt: skip
    text = lines(card)
    assert "Not scored: ABC, 1 call(s) from 2026-11-05 (no open price)." in text
    assert "Not scored: XYZ, 3 call(s) from 2026-11-05 to 2026-11-09 (no price downloaded)." in text


def test_a_finished_call_waiting_for_a_price_is_not_shown_as_due(tmp_path: Path) -> None:
    _save(tmp_path, _record())
    bars = _prices(15, 22)  # the exit has passed, the stock's price has a gap
    card = scorecard(tmp_path, EXIT + timedelta(days=4), prices=_fetcher(bars), save=False)
    assert card["next_due"] is None and card["waiting"] == 1
    assert "1 call(s) still open; 1 finished but waiting for a price." in lines(card)[1]


def test_weekly_runs_only_on_the_weeks_last_session(tmp_path: Path) -> None:
    def refuse(tickers: list[str], period: str) -> Prices:
        raise AssertionError("no fetch")

    assert weekly(tmp_path, date(2026, 11, 5), refuse) is None
    card = weekly(tmp_path, date(2026, 11, 6), refuse)
    assert card is not None and card["lines"][0].startswith("Ahead so far")


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
    assert record["scorecard"]["lines"][0].startswith("Ahead so far: no recommendation")
    [email] = outbox.sent
    assert email.subject.startswith("Weekly scorecard + tenq daily, session 2026-11-06")
    assert email.text.startswith("WEEKLY SCORECARD\nAhead so far")
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


def test_a_test_that_reaches_yahoo_fails_even_through_the_catch_all() -> None:
    from conftest import NoNetwork

    from tenq.prices import YahooPrices, fetch

    with pytest.raises(NoNetwork):
        fetch(["CRWD"], YahooPrices(), period="1mo")


def test_opus_lean_is_read_from_the_record_and_old_records_have_none() -> None:
    stock = _stock()
    stock["opus"]["lean"] = -0.25
    [made] = calls_from(_record("2026-11-05", MADE, stock))
    assert made.opus_lean == pytest.approx(-0.25)
    [old] = calls_from(_record())
    assert old.opus_lean is None
