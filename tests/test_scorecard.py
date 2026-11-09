from __future__ import annotations

import json
import math
import random
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from test_daily import Outbox, _ctx, _no_trace

import tenq.scorecard
from tenq.daily import main as daily_main
from tenq.daily_email import render
from tenq.prices import BENCHMARK, Bar, Prices
from tenq.scorecard import (
    _lead_lines,
    _luck_lines,
    _verdict,
    calls_from,
    daily_records,
    day_totals,
    last_session_of_week,
    left_out,
    lines,
    load_config,
    moved,
    next_session,
    on_picks,
    outcome,
    paired,
    planned_entry,
    planned_exit,
    rank_ic,
    ranking,
    read_ledger,
    scorecard,
    screen_groups,
    screen_ledger_path,
    screen_lines,
    screened_from,
    spearman,
    standard_error,
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
    assert standard_error(totals, 4, 1) == pytest.approx((11 * 4 / 3) ** 0.5 / 4)
    assert standard_error(totals, 4, 0) == pytest.approx((14 * 4 / 3) ** 0.5 / 4)
    assert standard_error(totals, 4, 2) == pytest.approx(
        ((14 + 2 * (2 / 3) * -3 + 2 * (1 / 3) * 2) * 4 / 3) ** 0.5 / 4
    )
    assert standard_error([1.0], 4, 1) is None


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


def _mixed_day(day: str, missing: int) -> list[dict[str, Any]]:
    """Six Opus Buys whose confidence orders the stocks against their excess
    returns while their leans order them with it; `missing` leans left out."""
    rows = [
        {"day": day, "opus": "Buy", "opus_confidence": 0.9 - i / 10, "opus_lean": 0.1 + i / 10}
        | {"excess": i}
        for i in range(6)
    ]
    for r in rows[2 : 2 + missing]:
        r["opus_lean"] = None
    return rows


def test_a_day_with_five_leans_ranks_those_by_lean_and_fewer_falls_back() -> None:
    assert rank_ic(_mixed_day("2026-11-05", 0), "opus", CFG) == pytest.approx(1.0)
    # one lean missing: the five that have one are still ranked by lean
    assert rank_ic(_mixed_day("2026-11-05", 1), "opus", CFG) == pytest.approx(1.0)
    # two missing leaves four: all six are ranked by weight times confidence
    assert rank_ic(_mixed_day("2026-11-05", 2), "opus", CFG) == pytest.approx(-1.0)


def test_each_day_picks_its_own_ranking_and_the_card_says_how_many_used_the_lean() -> None:
    rows = _mixed_day("2026-11-05", 0) + _mixed_day("2026-11-06", 2)
    assert ranking(rows, "opus", CFG) == {"ic": pytest.approx(0.0), "lean_days": 1, "days": 2}
    # Jev has no lean: its days are never counted as lean days
    jev = [r | {"jev_score": r["excess"]} for r in rows]
    assert ranking(jev, "jev", CFG)["lean_days"] == 0


@pytest.mark.parametrize(("kept", "by_lean"), [(16, True), (15, False)])
def test_most_of_a_days_calls_must_keep_a_lean_to_rank_by_lean(kept: int, by_lean: bool) -> None:
    # twenty Opus Buys: confidence orders them against their excess returns,
    # leans with them; 16 of 20 leans (80%) is enough, 15 is not
    rows = [
        {"day": "2026-11-05", "opus": "Buy", "opus_confidence": 0.99 - i / 100}
        | {"opus_lean": 0.01 + i / 100 if i < kept else None, "excess": i}
        for i in range(20)
    ]
    want = {"ic": pytest.approx(1.0 if by_lean else -1.0), "lean_days": int(by_lean), "days": 1}
    assert ranking(rows, "opus", CFG) == want


def test_a_day_of_tied_leans_falls_back_and_is_not_a_lean_day() -> None:
    rows = [r | {"opus_lean": 0.37} for r in _mixed_day("2026-11-05", 0)]
    assert ranking(rows, "opus", CFG) == {"ic": pytest.approx(-1.0), "lean_days": 0, "days": 1}


def test_a_stock_with_no_opus_call_is_not_ranked_by_lean() -> None:
    # five leaned Opus calls plus one stock Opus made no call on: the lean day
    # ranks the five only; the stray row's lean is never read
    stray = {"day": "2026-11-05", "opus": None, "opus_lean": 0.99, "excess": -5}
    rows = [*_mixed_day("2026-11-05", 1), stray]
    assert ranking(rows, "opus", CFG) == {"ic": pytest.approx(1.0), "lean_days": 1, "days": 1}


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


def test_the_opus_line_says_on_how_many_days_the_lean_ranked_it(tmp_path: Path) -> None:
    card = _card(tmp_path, _record())
    card["methods"]["opus"] |= {"rank_ic": 0.12, "ranked_days": 15, "lean_days": 12}
    card["methods"]["jev"] |= {"rank_ic": 0.05, "ranked_days": 15, "lean_days": 0}
    text = lines(card)
    opus = next(t for t in text if t.startswith("Opus: "))
    jev = next(t for t in text if t.startswith("Jev: "))
    assert "ranking +0.12 (by lean on 12 of 15 days);" in opus
    assert "by lean" not in jev
    # no ranked day yet: nothing to say about the lean
    card["methods"]["opus"] |= {"rank_ic": None, "ranked_days": 0, "lean_days": 0}
    assert not any("by lean" in t for t in lines(card))


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


def _friday(tmp_path: Path, score_prices: Any, outbox: Outbox, **kw: Any) -> dict[str, Any]:
    ctx = _ctx()
    ctx.now = FRIDAY
    code = daily_main(
        ["--runs", str(tmp_path), "--stage", "all"],
        ctx=ctx,
        **kw,
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


def test_a_failed_screen_comes_before_even_the_weekly_scorecard(tmp_path: Path) -> None:
    outbox = Outbox()
    failed = {"kind": "screen", "ok": False, "refused": ["no QQQ prices"]}
    _friday(tmp_path, _fetcher({}), outbox, screener=lambda now: failed)
    [email] = outbox.sent
    assert email.subject.startswith("SCREEN FAILED + Weekly scorecard + tenq daily")
    assert email.text.startswith("SCREEN FAILED: no QQQ prices.")
    assert email.text.index("SCREEN FAILED") < email.text.index("WEEKLY SCORECARD")
    assert email.html.index("#b00") < email.html.index("Weekly scorecard")


# --- Opus Shorts from before call-v10 (PR #24: Short meant "will fall outright") ----


def _old(*stocks: dict[str, Any], prompt: str = "call-v9") -> dict[str, Any]:
    return _record("2026-11-05", MADE, *stocks, versions={"call_prompt": prompt})


def test_an_old_opus_short_is_left_out_and_jevs_call_on_that_stock_kept() -> None:
    stock = _stock("TEAM", jev="Sell", opus="Short")
    stock["opus"]["lean"] = -0.5
    record = _old(stock)
    [made] = calls_from(record)
    assert made.jev == "Sell" and made.jev_score == pytest.approx(0.3)
    assert made.opus is None and made.opus_confidence is None and made.opus_lean is None
    assert left_out(record) == [
        {"day": "2026-11-05", "ticker": "TEAM", "method": "opus", "call": "Short",
         "prompt": "call-v9", "jev": "Sell"}
    ]  # fmt: skip


@pytest.mark.parametrize(
    ("prompt", "opus", "kept"),
    [
        ("call-v9", "Short", False),
        ("call-v1", "Short", False),
        ("call-v9", "Sell", True),  # only Short changed meaning
        ("call-v9", "Buy", True),
        ("call-v9", "Hold", True),
        ("call-v10", "Short", True),  # from call-v10 on, Short means "lag QQQ by a wide margin"
        ("call-v12", "Short", True),
        ("", "Short", True),  # a version that cannot be read is taken as current
        ("call-v9b", "Short", True),
    ],
)
def test_only_opus_shorts_from_before_call_v10_are_left_out(
    prompt: str, opus: str, kept: bool
) -> None:
    record = _old(_stock(opus=opus), prompt=prompt)
    [made] = calls_from(record)
    assert made.jev == "Buy"
    assert (made.opus == opus) is kept and (made.opus is None) is not kept
    assert (left_out(record) == []) is kept


def test_a_record_with_no_versions_keeps_its_shorts() -> None:
    record = _record(versions=None)
    [made] = calls_from(record)
    assert made.opus == "Short" and left_out(record) == []


def test_jevs_short_in_an_old_record_is_kept() -> None:
    # Jev's rules kept the same settings from rules-v7 to v8: its Short never changed
    record = _old(_stock("HUBS", jev="Short", opus="Sell"))
    [made] = calls_from(record)
    assert made.jev == "Short" and made.opus == "Sell" and left_out(record) == []


def test_a_stock_whose_only_call_was_an_old_short_is_not_scored() -> None:
    record = _old(_stock(jev=None, opus="Short"))
    assert calls_from(record) == []
    assert [(x["ticker"], x["jev"]) for x in left_out(record)] == [("CRWD", None)]


def test_the_card_scores_without_the_old_short_and_names_it(tmp_path: Path) -> None:
    bars = _prices() | {"DDOG": _prices()["CRWD"]}
    _save(tmp_path, _old(_stock(jev="Sell", opus="Short"), _stock("DDOG", jev="Buy", opus="Sell")))
    card = scorecard(tmp_path, EXIT, CFG, _fetcher(bars), save=False)
    assert card["scored"] == 2
    assert card["methods"]["jev"]["calls"] == 2
    assert card["methods"]["opus"]["calls"] == 1
    assert card["methods"]["opus"]["counts"] == {"Buy": 0, "Hold": 0, "Sell": 1, "Short": 0}
    assert card["paired"]["pairs"] == 1  # CRWD drops out of Jev against Opus
    assert card["call_prompts"] == {"call-v9": 1}  # only the Opus call that was scored
    assert card["left_out"] == left_out(_old(_stock(jev="Sell", opus="Short")))
    assert (
        "Left out: Opus's Short on CRWD from 2026-11-05. Its prompt then (call-v9) said "
        "Short meant the stock would fall outright; since call-v10, Short means it will "
        "lag QQQ by a wide margin, which is what the card scores. Jev's Sell on CRWD "
        "that day still counts."
    ) in lines(card)


def test_a_card_with_nothing_left_out_says_nothing_about_it(tmp_path: Path) -> None:
    card = _card(tmp_path, _record())
    assert card["left_out"] == []
    assert not any(t.startswith("Left out:") for t in lines(card))


def test_fridays_email_names_the_old_shorts_before_anything_scores(tmp_path: Path) -> None:
    _save(
        tmp_path,
        _old(_stock("TEAM", jev="Sell", opus="Short"), _stock("HUBS", jev=None, opus="Short")),
    )
    outbox = Outbox()
    record = _friday(tmp_path, _fetcher({}), outbox)
    assert record["scorecard"]["scored"] == 0
    [email] = outbox.sent
    weekly = email.text[: email.text.index("tenq daily")]
    [team, hubs] = [t for t in weekly.splitlines() if t.startswith("Left out:")]
    assert team.startswith("Left out: Opus's Short on TEAM from 2026-11-05.")
    assert team.endswith("Jev's Sell on TEAM that day still counts.")
    assert hubs.startswith("Left out: Opus's Short on HUBS from 2026-11-05.")
    assert hubs.endswith("which is what the card scores.")  # Jev made no call on HUBS
    assert "Short on TEAM from 2026-11-05" in email.html


# --- the screen checks ----------------------------------------------------------


def _screen(
    chosen: tuple[str, ...] = ("PICK",),
    over_cap: tuple[str, ...] = ("CUT",),
    filled: tuple[str, ...] = ("DRAW",),
    held: tuple[str, ...] = ("HELD", "CALM"),
    unscreened: tuple[str, ...] = ("LOST",),
    used: str = "screen",
) -> dict[str, Any]:
    def row(t: str) -> dict[str, Any]:
        return {"ticker": t, "reasons": ["a new 8-K"]}

    return {
        "kind": "screen",
        "ok": True,
        "version": "screen-v6",
        "day": "2026-11-05",
        "chosen": [row(t) for t in chosen],
        "over_cap": [row(t) for t in over_cap],
        "filled": list(filled),
        "held": list(held),
        "unscreened": list(unscreened),
        "used": used,
    }


def _screened(**kw: Any) -> dict[str, Any]:
    """A screened day: both methods called the pick and the draw, Buy and Short."""
    return _record("2026-11-05", MADE, _stock("PICK"), _stock("DRAW"), screen=_screen(**kw))


# each stock's 20-session return; QQQ makes +2%, so the excess is 2 points less
RETURNS = {"PICK": 0.12, "CUT": -0.08, "DRAW": 0.04, "HELD": -0.02, "CALM": 0.02, "LOST": 0.5}


def _many(returns: dict[str, float] = RETURNS) -> dict[str, tuple[Bar, ...]]:
    out = {t: _bars(_sessions(20), 100.0, 101.0, 100.0 * (1 + r)) for t, r in returns.items()}
    return out | {BENCHMARK: _bars(_sessions(20), 200.0, 201.0, 204.0)}


def _screen_card(runs: Path, fetch: Any = None, **kw: Any) -> dict[str, Any]:
    return scorecard(runs, EXIT, CFG, fetch or _fetcher(_many()), **kw)


def test_each_screened_stock_gets_its_group_and_unscreened_ones_get_none() -> None:
    assert screen_groups(_screened()) == {
        "PICK": "picked",
        "CUT": "over_cap",
        "DRAW": "random",
        "HELD": "held",
        "CALM": "held",
    }
    # a stock listed twice (never by the screen itself) takes its first group
    assert screen_groups(_screened(held=("HELD", "PICK")))["PICK"] == "picked"


@pytest.mark.parametrize(
    "record",
    [
        _record(),  # before the screen
        _record(screen=_screen(used="watchlist")),  # the screen failed: the watchlist ran
        _record(screen=None),
    ],
)
def test_a_day_the_screen_did_not_choose_has_no_groups(record: dict[str, Any]) -> None:
    assert screen_groups(record) == {}
    assert screened_from(record) == []
    assert {m.group for m in calls_from(record)} == {None}


def test_calls_carry_the_group_and_check_1_gets_every_screened_stock() -> None:
    record = _screened()
    assert {m.ticker: m.group for m in calls_from(record)} == {"PICK": "picked", "DRAW": "random"}
    everyone = screened_from(record)
    assert [(m.ticker, m.group) for m in everyone] == [
        ("CALM", "held"),
        ("CUT", "over_cap"),
        ("DRAW", "random"),
        ("HELD", "held"),
        ("PICK", "picked"),
    ]
    assert {m.made_at for m in everyone} == {MADE}  # the same window as the calls
    assert {m.jev for m in everyone} | {m.opus for m in everyone} == {None}
    assert everyone[0].versions == {"screen": "screen-v6"}


def test_check_1_sets_the_size_of_moves_that_passed_against_those_that_did_not(
    tmp_path: Path,
) -> None:
    _save(tmp_path, _screened())
    mv = _screen_card(tmp_path, save=False)["screen"]["moves"]
    # excess: PICK +10%, CUT -10% passed; DRAW 2%, HELD -4%, CALM 0% did not
    assert (mv["passed"], mv["not_passed"]) == (2, 3)
    assert mv["passed_mean"] == pytest.approx(0.10)
    assert mv["not_passed_mean"] == pytest.approx(0.02)
    assert mv["gap"] == pytest.approx(0.08)
    assert mv["days"] == 1
    # the luck range: any 2 of the 5 could have passed; 1 pair in 10 gives +8.00%, 1 gives -7.00%
    assert round(mv["luck"], 4) in (0.07, 0.08)


def test_check_2_sets_each_methods_calls_on_the_picks_against_the_draw(tmp_path: Path) -> None:
    _save(tmp_path, _screened())
    c = _screen_card(tmp_path, save=False)["screen"]["calls"]
    # Jev bought both: PICK +10%, DRAW +2%; Opus shorted both, weight -1
    assert (c["jev"]["picks"], c["jev"]["draw"]) == (1, 1)
    assert c["jev"]["picks_mean"] == pytest.approx(0.10)
    assert c["jev"]["draw_mean"] == pytest.approx(0.02)
    assert c["jev"]["gap"] == pytest.approx(0.08)
    assert c["opus"]["gap"] == pytest.approx(-0.08)


def test_a_watchlist_day_counts_in_neither_check(tmp_path: Path) -> None:
    _save(tmp_path, _record("2026-11-05", MADE, _stock("PICK"), _stock("DRAW")))
    sc = _screen_card(tmp_path, save=False)["screen"]
    assert sc["moves"] == {
        "passed": 0,
        "not_passed": 0,
        "days": 0,
        "passed_mean": None,
        "not_passed_mean": None,
    }
    assert sc["calls"]["jev"]["picks"] == sc["calls"]["jev"]["draw"] == 0
    assert sc["open"] == 0 and sc["biggest_held"] == []


def test_the_biggest_held_moves_are_the_ones_finished_this_time(tmp_path: Path) -> None:
    _save(tmp_path, _screened(held=("HELD", "CALM", "PICK2")))
    bars = _many(RETURNS | {"PICK2": 0.30, "HELD": -0.12})
    sc = _screen_card(tmp_path, _fetcher(bars))["screen"]
    assert [(x["ticker"], round(x["excess"], 4)) for x in sc["biggest_held"]] == [
        ("PICK2", 0.28),
        ("HELD", -0.14),
        ("CALM", 0.0),
    ]
    again = _screen_card(tmp_path, _fetcher(bars))["screen"]
    assert again["biggest_held"] == []  # already in the ledger: not news next week
    assert again["moves"]["not_passed"] == 4


def test_the_screen_ledger_is_one_row_a_line_and_read_back_without_a_new_fetch(
    tmp_path: Path,
) -> None:
    _save(tmp_path, _screened())
    _screen_card(tmp_path)
    path = screen_ledger_path(tmp_path, CFG)
    assert path.name == f"{CFG.version}-screen.json"
    text = path.read_text()
    assert len(text.splitlines()) == 2 + 5  # head, 5 rows, closing braces
    rows = read_ledger(path, CFG)
    assert rows["2026-11-05:CUT"] == {
        "day": "2026-11-05",
        "made_at": MADE.isoformat(),
        "ticker": "CUT",
        "group": "over_cap",
        "versions": {"screen": "screen-v6"},
        "status": "scored",
        "reason": "",
        "entry_day": ENTRY.isoformat(),
        "exit_day": EXIT.isoformat(),
        "excess": pytest.approx(-0.10),
        "score_version": CFG.version,
    }
    asked: list[Any] = []
    sc = _screen_card(tmp_path, _fetcher({}, asked))["screen"]
    assert asked == [] and sc["moves"]["gap"] == pytest.approx(0.08)


def test_call_rows_in_the_ledger_keep_their_group(tmp_path: Path) -> None:
    _save(tmp_path, _screened())
    _screen_card(tmp_path)
    [row] = [
        r
        for k, r in read_ledger(tmp_path / "daily" / "scores" / f"{CFG.version}.json", CFG).items()
        if k.endswith("DRAW")
    ]
    assert row["group"] == "random"


def test_a_failed_screen_fetch_loses_check_1_only(tmp_path: Path) -> None:
    _save(tmp_path, _screened())

    def fetch(tickers: list[str], period: str) -> Prices:
        if "HELD" in tickers:
            raise RuntimeError("Yahoo is down")
        return Prices(_many(), ())

    card = _screen_card(tmp_path, fetch)
    assert card["scored"] == 2  # the calls are scored
    assert card["screen"]["error"] == "check 1 could not be made (RuntimeError)"
    assert card["screen"]["calls"]["jev"]["gap"] == pytest.approx(0.08)
    said = lines(card)
    assert (
        "Screen check 1 was not made this week: check 1 could not be made (RuntimeError)." in said
    )
    assert any(t.startswith("Screen check 2, are Jev's calls better") for t in said)
    assert not screen_ledger_path(tmp_path, CFG).exists()


def test_before_any_screened_stock_finishes_the_card_says_when_the_first_will(
    tmp_path: Path,
) -> None:
    _save(tmp_path, _screened())
    card = scorecard(tmp_path, EXIT - timedelta(days=1), CFG, _fetcher({}), save=False)
    assert card["screen"]["finished"] == 0
    said = [t for t in lines(card) if t.startswith("Screen")]
    assert said == [
        "Screen checks: no screened stock has finished its 20 sessions yet "
        f"(5 stock-days open; the first finishes on {EXIT})."
    ]


def test_a_card_with_no_screened_day_says_nothing_of_the_screen(tmp_path: Path) -> None:
    _save(tmp_path, _record())
    card = _card(tmp_path, today=EXIT)
    assert not [t for t in lines(card) if "creen" in t]


def test_the_card_words_both_checks(tmp_path: Path) -> None:
    _save(tmp_path, _screened())
    said = lines(_screen_card(tmp_path, save=False))
    said = said[said.index(next(t for t in said if t.startswith("Screen"))) : -1]
    assert said == [
        "Screen check 1, do the stocks that pass move more: over 20 sessions, up or down "
        "against QQQ, those that passed moved 10.00% on average (2 stock-days), those that "
        "did not 2.00% (3), comparing stocks screened the same day (1 day).",
        "Gap (passed less not passed): +8.00%. Not settled (too early: 1 of 13 weeks "
        "scored): 9 times in 10, luck alone gives a gap between -7.00% and +7.00%.",
        "The gap is outside that range, but since the card is read every week a gap settles "
        "only at 10.31%, and only once 13 weeks are scored.",
        "Screen check 2, are Jev's calls better on the screen's picks: +10.00% a call on the "
        "picks (1), +2.00% on the random draw (1), comparing picks and draws from the same "
        "day (1 day).",
        "Gap (picks less random draw): +8.00%. Not settled (too early: 1 of 13 weeks "
        "scored): 9 times in 10, luck alone gives a gap between -13.79% and +13.79%.",
        "Screen check 2, are Opus's calls better on the screen's picks: -10.00% a call on the "
        "picks (1), -2.00% on the random draw (1), comparing picks and draws from the same "
        "day (1 day).",
        "Gap (picks less random draw): -8.00%. Not settled (too early: 1 of 13 weeks "
        "scored): 9 times in 10, luck alone gives a gap between -13.79% and +13.79%.",
        "Biggest moves against QQQ among stocks Hold by screen, newly scored: HELD -4.00% "
        "(screened 2026-11-05), CALM +0.00% (screened 2026-11-05).",
    ]


def test_a_stock_day_with_no_price_is_named_and_left_out_of_check_1(tmp_path: Path) -> None:
    _save(tmp_path, _screened())
    bars = _many()
    del bars["CALM"]
    late = planned_exit(ENTRY, 20 + CFG.no_download_after)
    bars[BENCHMARK] = _bars(_sessions(20 + CFG.no_download_after), 200.0, 201.0, 204.0)
    card = scorecard(tmp_path, late, CFG, _fetcher(bars), save=False)
    assert card["screen"]["unscored"] == 1 and card["screen"]["moves"]["not_passed"] == 2
    assert card["screen"]["finished"] == 5  # the stock-day with no price finished too
    assert "Screen check 1 leaves out 1 stock-day(s) with no price." in lines(card)


def _run_rows(
    sizes: dict[str, list[float]], passed: dict[str, list[bool | None]]
) -> list[dict[str, Any]]:
    """Check 1 rows: each stock's move day by day (every other one downward,
    since only the size counts) and whether it passed the screen that day."""
    days = [d.isoformat() for d in _sessions(len(next(iter(sizes.values()))))]
    return [
        {
            "day": day,
            "ticker": t,
            "status": "scored",
            "group": "picked" if ok else "held",
            "excess": -x if i % 2 else x,
        }
        for t, xs in sizes.items()
        for i, (day, x, ok) in enumerate(zip(days, xs, passed[t], strict=True))
        if ok is not None
    ]


def test_check_1_sets_each_days_stocks_only_against_that_days() -> None:
    # no skill: on the busy day every stock moves 10% and five of seven pass;
    # on the quiet day every stock moves 4% and one passes. Pooled, the stocks
    # that passed would look 3.5 points wilder, only because of the busy day.
    sizes = {f"S{i}": [0.10, 0.04] for i in range(7)}
    passed: dict[str, list[bool | None]] = {f"S{i}": [i < 5, i < 1] for i in range(7)}
    mv = moved(_run_rows(sizes, passed), CFG)
    assert (mv["passed"], mv["not_passed"], mv["days"]) == (6, 8, 2)
    assert mv["passed_mean"] == pytest.approx(0.07)
    assert mv["not_passed_mean"] == pytest.approx(0.07)
    assert mv["gap"] == pytest.approx(0.0)


def test_a_day_with_no_stock_on_one_side_sits_out_check_1() -> None:
    sizes = {"A": [0.10, 0.30], "B": [0.02, 0.01]}
    passed: dict[str, list[bool | None]] = {"A": [True, True], "B": [False, None]}
    mv = moved(_run_rows(sizes, passed), CFG)
    assert (mv["passed"], mv["not_passed"], mv["days"]) == (1, 1, 1)
    assert mv["gap"] == pytest.approx(0.08)


def test_check_1s_luck_hands_each_stocks_whole_run_of_results_to_another_stock() -> None:
    # A passes every day and B never does, so a shuffle either keeps them or
    # swaps them: the gap luck gives is this gap, one way round or the other.
    # Shuffling single days instead would mix the days and widen the range.
    sizes = {"A": [0.30, 0.02, 0.10, 0.06], "B": [0.01, 0.05, 0.02, 0.04]}
    passed: dict[str, list[bool | None]] = {"A": [True] * 4, "B": [False] * 4}
    mv = moved(_run_rows(sizes, passed), CFG)
    assert mv["gap"] == pytest.approx(0.09)  # (0.29 - 0.03 + 0.08 + 0.02) / 4
    assert mv["luck"] == pytest.approx(0.09) and mv["se"] == pytest.approx(0.09)


def test_check_1s_luck_is_the_gap_9_shuffles_in_10_stay_within() -> None:
    # one day, one stock passed: A or B passing gives +4.50%, C passing -9.00%,
    # so 1 shuffle in 3 goes past 4.50% and the range reaches 9.00%
    rows = _run_rows(
        {"A": [0.10], "B": [0.10], "C": [0.01]}, {"A": [True], "B": [False], "C": [False]}
    )
    mv = moved(rows, CFG)
    assert mv["gap"] == pytest.approx(0.045)
    assert mv["luck"] == pytest.approx(0.09)
    se = (2 / 3 * 0.045**2 + 1 / 3 * 0.09**2) ** 0.5
    assert mv["se"] == pytest.approx(se, rel=0.05)
    assert moved(rows, CFG) == mv  # seeded: the same card when made again


def test_check_2_sets_each_days_picks_only_against_that_days_draw() -> None:
    def row(day: date, t: str, group: str, excess: float) -> dict[str, Any]:
        return {
            "day": day.isoformat(),
            "ticker": t,
            "status": "scored",
            "group": group,
            "jev": "Buy",
            "opus": None,
            "excess": excess,
            "exit_day": next_session(day).isoformat(),
        }

    one, two, three, four = _sessions(4)
    rows = [
        row(one, "P1", "picked", 0.10),
        row(one, "P2", "picked", 0.06),
        row(one, "R1", "random", 0.06),
        row(two, "P3", "picked", -0.04),
        row(two, "R2", "random", 0.00),
        row(two, "R3", "random", 0.02),
        row(two, "R4", "random", 0.04),
        row(three, "P4", "picked", 0.50),  # no draw that day: it sits out
        row(four, "R5", "random", 0.90),  # nothing passed that day: no picks, it sits out
    ]
    c = on_picks(rows, CFG)
    jev = c["jev"]
    # day one +8.00% against +6.00%, day two -4.00% against +2.00%; pooled,
    # the picks would make +4.00% against +3.00%
    assert (jev["picks"], jev["draw"], jev["days"]) == (3, 4, 2)
    assert (jev["no_draw"], jev["no_picks"]) == (1, 1)
    assert jev["picks_mean"] == pytest.approx(0.02)
    assert jev["draw_mean"] == pytest.approx(0.04)
    assert jev["gap"] == pytest.approx(-0.02)
    assert c["opus"]["days"] == 0 and c["opus"]["picks_mean"] is None


def test_check_2s_luck_range_comes_from_the_daily_moves() -> None:
    rows: list[dict[str, Any]] = []
    for i, d in enumerate(_sessions(12)):
        for t, group, call, size, shift in (
            ("A", "picked", "Buy", 1.0, 0),
            ("B", "random", "Short", -0.5, 1),  # not a mirror of A: the gap varies by day
        ):
            moves = [0.01 * size * ((i + j + shift) % 3 - 1) for j in range(3)]
            rows.append(
                {
                    "day": d.isoformat(),
                    "ticker": t,
                    "status": "scored",
                    "group": group,
                    "jev": call,
                    "opus": "Hold",
                    "excess": sum(moves),
                    "moves": moves,
                    "move_days": [x.isoformat() for x in _sessions(3, d)],
                }
            )
    c = on_picks(rows, CFG)["jev"]
    w = CFG.weights
    # one pick and one draw a day: each weighs 1 / 12 of the gap
    parts = [((1 if r["group"] == "picked" else -1) * w[r["jev"]] / 12, r) for r in rows]
    totals = list(day_totals(parts).values())
    se = standard_error(totals, 1, CFG.carry_days)
    assert se  # a real spread, not zero
    assert c["days"] == 12 and len(totals) == 14  # spread over the move days
    assert c["se"] == pytest.approx(se) and c["luck"] == pytest.approx(t95(13) * se)
    assert c["gap"] == pytest.approx(c["picks_mean"] - c["draw_mean"])
    assert on_picks(rows, CFG)["opus"]["gap"] == pytest.approx(0.0)  # Hold scores 0


@pytest.mark.parametrize(
    ("gap", "weeks", "settled"),
    [
        (0.05, 13, True),
        (-0.05, 13, True),
        (0.05, 12, False),  # too early, however big
        (0.01, 13, False),  # under the settle bar
    ],
)
def test_a_screen_gap_settles_only_after_13_weeks_and_past_the_bar(
    gap: float, weeks: int, settled: bool
) -> None:
    g = _verdict(gap, 0.01, 0.0165, weeks, CFG)
    assert g["bar"] == pytest.approx(CFG.settled_bar * 0.01)  # 2.40%
    assert g["settled"] is settled


def test_a_screen_gap_that_rounds_to_zero_never_settles_however_steady() -> None:
    assert _verdict(0.00004, 1e-7, 1.6e-7, 13, CFG)["settled"] is False  # shows as +0.00%
    assert _verdict(0.0001, 1e-7, 1.6e-7, 13, CFG)["settled"] is True  # shows as +0.01%
    assert _verdict(0.05, None, None, 13, CFG)["settled"] is False  # luck not sized


def test_the_settled_line_says_which_way() -> None:
    g = {"gap": 0.03, "weeks": 13, "luck": 0.01, "bar": 0.015, "settled": True}
    up = _luck_lines(g, 13, "a less b", "they move more", "they move less")
    assert up == [
        "Gap (a less b): +3.00%. Settled: they move more (a gap this size is unlikely to be "
        "luck, even counting every weekly check)."
    ]
    down = _luck_lines(g | {"gap": -0.03}, 13, "a less b", "they move more", "they move less")
    assert "Settled: they move less" in down[0]


def test_a_gap_outside_the_luck_range_that_has_not_settled_says_why() -> None:
    g = {"gap": 0.03, "weeks": 13, "luck": 0.01, "bar": 0.04, "settled": False}
    head = (
        "Gap (a less b): +3.00%. Not settled: 9 times in 10, luck alone gives a gap between "
        "-1.00% and +1.00%."
    )
    assert _luck_lines(g, 13, "a less b", "x", "y") == [
        head,
        "The gap is outside that range, but since the card is read every week a gap settles "
        "only at 4.00%.",
    ]
    early = _luck_lines(g | {"weeks": 2}, 13, "a less b", "x", "y")
    assert early[1].endswith("settles only at 4.00%, and only once 13 weeks are scored.")
    clears = _luck_lines(g | {"weeks": 2, "bar": 0.02}, 13, "a less b", "x", "y")
    assert clears[1] == (
        "The gap already clears the settle bar of 2.00%; a gap can settle only once 13 weeks "
        "are scored."
    )
    assert _luck_lines(g | {"gap": 0.005}, 13, "a less b", "x", "y") == [
        head.replace("+3.00%", "+0.50%")
    ]


def test_screened_stocks_waiting_for_a_price_are_counted_not_called_unfinished(
    tmp_path: Path,
) -> None:
    _save(tmp_path, _screened())
    qqq = {BENCHMARK: _many()[BENCHMARK]}  # QQQ's prices came, the stocks' did not

    def fetch(tickers: list[str], period: str) -> Prices:
        return Prices(qqq, ("price download failed (timeout)",))

    card = _screen_card(tmp_path, fetch, save=False)
    assert card["screen"]["waiting"] == 5 and card["screen"]["moves"]["days"] == 0
    said = [t for t in lines(card) if t.startswith("Screen")]
    assert said == ["Screen check 1: 5 stock-day(s) finished but are waiting for a price."]


def test_a_screen_list_that_is_not_a_list_is_skipped() -> None:
    record = _screened()
    record["screen"]["held"] = 5
    record["screen"]["filled"] = "DRAW"  # not read letter by letter
    assert screen_groups(record) == {"PICK": "picked", "CUT": "over_cap"}


def test_a_day_with_no_random_draw_sits_out_check_2_and_the_card_says_so(
    tmp_path: Path,
) -> None:
    _save(tmp_path, _screened(filled=()))
    card = _screen_card(tmp_path, save=False)
    jev = card["screen"]["calls"]["jev"]
    assert (jev["days"], jev["no_draw"], jev["no_picks"], jev["picks_mean"]) == (0, 1, 0, None)
    assert (
        "Screen check 2 for Jev's calls: no finished day had both picks and a random draw "
        "yet. 1 day with no random draw sat out."
    ) in lines(card)


def test_check_2s_luck_weighs_each_stock_by_its_own_days_group_size() -> None:
    # day one: 1 pick against 3 draws; day two: 2 picks against 1 draw. Each
    # day counts the same, so a stock's weight is 1 / (its group that day x 2),
    # and the daily moves add up to the gap exactly.
    one, two = _sessions(2)
    stocks = (
        (one, "P1", "picked", "Buy", [0.03, -0.01, 0.02]),
        (one, "R1", "random", "Buy", [0.01, 0.00, -0.02]),
        (one, "R2", "random", "Sell", [-0.02, 0.01, 0.00]),
        (one, "R3", "random", "Buy", [0.00, 0.02, 0.01]),
        (two, "P2", "picked", "Buy", [0.02, 0.01, -0.03]),
        (two, "P3", "picked", "Short", [-0.01, -0.02, 0.01]),
        (two, "R4", "random", "Buy", [0.01, -0.01, 0.00]),
    )
    rows = [
        {
            "day": d.isoformat(),
            "ticker": t,
            "status": "scored",
            "group": g,
            "jev": call,
            "opus": None,
            "excess": sum(moves),
            "moves": moves,
            "move_days": [x.isoformat() for x in _sessions(3, d)],
        }
        for d, t, g, call, moves in stocks
    ]
    w = CFG.weights
    size = {(one, "picked"): 1, (one, "random"): 3, (two, "picked"): 2, (two, "random"): 1}
    parts = [
        ((1 if g == "picked" else -1) * w[call] / (size[d, g] * 2), r)
        for (d, _, g, call, _), r in zip(stocks, rows, strict=True)
    ]
    totals = list(day_totals(parts).values())
    c = on_picks(rows, CFG)["jev"]
    assert math.isclose(sum(totals), c["gap"])  # the parts are the gap, day by day
    assert c["se"] == pytest.approx(standard_error(totals, 1, CFG.carry_days))


def test_the_weeks_count_only_days_both_groups_were_there() -> None:
    # three weeks screened; in the middle one only a stock that passed was
    # priced, so that week did not compare anything
    picked = [True, None, None, None, None, True, None, None, None, None, True]
    held = [False, None, None, None, None, None, None, None, None, None, False]
    sizes = {"A": [0.05] * 11, "B": [0.02] * 11}
    mv = moved(_run_rows(sizes, {"A": picked, "B": held}), CFG)
    assert (mv["days"], mv["weeks"]) == (2, 2)
    rows = [
        {
            "day": d.isoformat(),
            "ticker": t,
            "status": "scored",
            "group": g,
            "jev": "Buy",
            "opus": None,
            "excess": 0.01,
            "moves": [0.01],
            "move_days": [d.isoformat()],
        }
        for i, d in enumerate(_sessions(11))
        for t, g, on in (("P", "picked", (0, 5, 10)), ("R", "random", (0, 10)))
        if i in on
    ]
    c = on_picks(rows, CFG)["jev"]
    assert (c["days"], c["weeks"]) == (2, 2)


def test_check_1s_luck_range_is_the_90th_percentile_of_the_shuffled_gaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[float, float, int, int, list[int]] | None] = []
    real = tenq.scorecard._size_gap

    def spy(*a: Any) -> tuple[float, float, int, int, list[int]] | None:
        got = real(*a)
        seen.append(got)
        return got

    monkeypatch.setattr(tenq.scorecard, "_size_gap", spy)
    sizes = {f"S{i}": [0.01 * (i + 1), 0.013 * (10 - i), 0.007 * (i + 3)] for i in range(10)}
    passed: dict[str, list[bool | None]] = {
        f"S{i}": [i % 3 == 0, i % 4 == 1, i < 4] for i in range(10)
    }
    mv = moved(_run_rows(sizes, passed), CFG, shuffles=20)
    gaps = [g[0] - g[1] for g in seen[1:] if g is not None]
    wide = sorted(abs(g) for g in gaps)
    assert len(gaps) == 20 and wide[16] < wide[17] < wide[18]  # the 85th, 90th, 95th differ
    assert mv["luck"] == pytest.approx(wide[17])
    assert mv["se"] == pytest.approx(math.sqrt(sum(g * g for g in gaps) / len(gaps)))


def test_check_2_says_which_days_sat_out_and_why() -> None:
    sc = {"calls": {"jev": {"days": 0, "no_draw": 2, "no_picks": 1}}}
    assert screen_lines(sc, CFG.weeks_first_verdict) == [
        "Screen check 2 for Jev's calls: no finished day had both picks and a random draw "
        "yet. 2 days with no random draw and 1 day with no picks sat out."
    ]


def test_finished_stocks_with_no_price_are_not_called_unfinished() -> None:
    sc = {
        "moves": {"days": 0},
        "finished": 2,
        "open": 5,
        "waiting": 0,
        "unscored": 2,
        "calls": {},
    }
    got = screen_lines(sc, CFG.weeks_first_verdict)
    assert not any("no screened stock has finished" in x for x in got)
    assert got == [
        "Screen check 1: no finished day has priced stocks both among those that passed "
        "and those that did not yet.",
        "Screen check 1 leaves out 2 stock-day(s) with no price.",
    ]
