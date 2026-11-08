"""The daily screen: which stocks go to Jev and Opus, decided by code alone."""

from __future__ import annotations

import itertools
import json
import math
import random
import statistics
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from tenq.config import load_watchlist
from tenq.filings import FilingMeta
from tenq.prices import BENCHMARK, Bar, Prices
from tenq.screen import (
    lines,
    load_config,
    load_universe,
    main,
    moves,
    previous_session,
    random_fill,
    screen,
)

DAY = date(2026, 11, 6)  # a Friday session; the previous session is 2026-11-05
CFG = load_config()


def _days(n: int, last: date = DAY) -> list[date]:
    out: list[date] = []
    d = last
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return out[::-1]


def _bars(jump: float = 0.0, n: int = 40) -> tuple[Bar, ...]:
    """Closes that swing 1% up and down each day, then move `jump` on the last."""
    days = _days(n)
    closes = [100.0 if i % 2 == 0 else 101.0 for i in range(n - 1)]
    closes.append(closes[-1] * (1 + jump))
    return tuple(Bar(d, c, 1e6) for d, c in zip(days, closes, strict=True))


def _flat(n: int = 40) -> tuple[Bar, ...]:
    return tuple(Bar(d, 500.0, 1e6) for d in _days(n))


def _prices(**jumps: float) -> Prices:
    bars = {BENCHMARK: _flat()} | {t: _bars(j) for t, j in jumps.items()}
    return Prices(bars, ())


def _meta(ticker: str, form: str, filed: date) -> FilingMeta:
    return FilingMeta(
        ticker=ticker,
        cik=1,
        company=ticker,
        form=form,
        accession=f"0000000001-26-{abs(hash((ticker, form, filed))) % 999999:06d}",
        filed=filed.isoformat(),
        period=filed.isoformat(),
        url="https://www.sec.gov/",
    )


class Filings:
    def __init__(
        self, listed: dict[str, list[FilingMeta]], broken: frozenset[str] = frozenset()
    ) -> None:
        self.listed, self.broken, self.asked = listed, broken, 0

    def list_recent(self, ticker: str, forms: tuple[str, ...], limit: int) -> list[FilingMeta]:
        self.asked += 1
        assert forms == CFG.forms  # the settings decide which forms are listed
        if ticker in self.broken:
            raise ConnectionError("EDGAR down")
        return self.listed.get(ticker, [])


# settings


def test_the_settings_load() -> None:
    assert CFG.version == "screen-v2"
    assert (CFG.move_sigma, CFG.cap, CFG.fill_to) == (2.5, 25, 20)
    assert CFG.forms == ("8-K", "10-Q", "10-K", "6-K", "20-F", "40-F")
    assert len(CFG.sha256) == 64


@pytest.mark.parametrize(
    ("edit", "says"),
    [
        (lambda t: t.replace("cap: 25", "cap: 25\nrecent_sessions: 20"), "expected exactly"),
        (lambda t: t.replace("fill_to: 20", "fill_to: 30"), "fill_to"),
        (lambda t: t.replace("fill_to: 20", "fill_to: -1"), "fill_to"),
        (lambda t: t.replace("cap: 25", "cap: true"), "cap"),
        (lambda t: t.replace("move_sigma: 2.5", "move_sigma: 0"), "move_sigma"),
        (lambda t: t.replace("forms: [8-K, 10-Q, 10-K, 6-K, 20-F, 40-F]", "forms: []"), "forms"),
    ],
)
def test_bad_settings_are_refused(tmp_path: Path, edit: object, says: str) -> None:
    text = (Path(__file__).parents[1] / "config" / "screen.yaml").read_text()
    path = tmp_path / "screen.yaml"
    path.write_text(edit(text))  # type: ignore[operator]
    with pytest.raises(ValueError, match=says):
        load_config(path)


def test_the_universe_is_the_watchlist_first_then_the_nasdaq_100_once_each() -> None:
    tickers = load_universe()
    watch = load_watchlist().tickers
    assert tickers[: len(watch)] == tuple(watch)
    assert len(tickers) == len(set(tickers)) == 113
    assert "GOOGL" in tickers and "GOOG" not in tickers


@pytest.mark.parametrize(
    ("day", "before"),
    [
        (date(2026, 11, 6), date(2026, 11, 5)),
        (date(2026, 11, 9), date(2026, 11, 6)),  # Monday: back over the weekend
        (date(2026, 11, 27), date(2026, 11, 25)),  # after Thanksgiving
    ],
)
def test_the_previous_session_skips_weekends_and_holidays(day: date, before: date) -> None:
    assert previous_session(day) == before


# the big-move reason


def test_a_big_move_is_measured_in_the_stocks_own_usual_moves() -> None:
    prices = _prices(BIG=0.08, DOWN=-0.08, QUIET=0.002)
    z, window, note = moves("BIG", prices, DAY)
    assert z is not None and z >= CFG.move_sigma and note is None
    assert window in (1, 5)
    z_down, _, _ = moves("DOWN", prices, DAY)
    assert z_down is not None and z_down <= -CFG.move_sigma
    z_quiet, _, _ = moves("QUIET", prices, DAY)
    assert z_quiet is not None and abs(z_quiet) < CFG.move_sigma


def test_a_stock_without_prices_is_noted_not_dropped_silently() -> None:
    assert moves("NONE", _prices(), DAY) == (None, None, "no prices")


def test_prices_that_end_before_the_session_are_not_used() -> None:
    """A lagging feed would otherwise pass a stock today on yesterday's move."""
    yesterday = date(2026, 11, 5)
    late = tuple(
        Bar(d, b.close, 1e6) for d, b in zip(_days(40, yesterday), _bars(0.08), strict=True)
    )
    prices = Prices({BENCHMARK: _flat(), "LATE": late}, ())
    assert moves("LATE", prices, DAY) == (None, None, "prices end 2026-11-05, not 2026-11-06")
    out = screen(["LATE"], DAY, DAY, CFG, prices, None)
    assert out["chosen"] == [] and out["unscreened"] == ["LATE"]


def test_a_steady_five_session_move_is_scaled_by_root_five() -> None:
    days = _days(40)
    closes = [100.0 if i % 2 == 0 else 101.0 for i in range(35)]
    for _ in range(5):
        closes.append(closes[-1] * 1.025)
    bars = tuple(Bar(d, c, 1e6) for d, c in zip(days, closes, strict=True))
    prices = Prices({BENCHMARK: _flat(), "STEADY": bars}, ())
    z, window, _ = moves("STEADY", prices, DAY)
    # QQQ is flat, so the daily moves against it are the stock's own; the
    # yardstick is the 20 sessions before the 5 tested, all the 1% swing.
    before = closes[-26:-5]
    usual = statistics.stdev(b / a - 1 for a, b in itertools.pairwise(before))
    assert window == 5
    assert z == pytest.approx((closes[-1] / closes[-6] - 1) / (usual * math.sqrt(5)))
    assert z is not None and z >= CFG.move_sigma


def _tracking(gap: float, n: int = 40) -> Prices:
    """QQQ swings 2% a day; TRACK follows it with a 0.3% wobble of its own,
    then on the last day moves `gap` more than QQQ."""
    days = _days(n)
    q, s = [500.0], [100.0]
    for i in range(1, n):
        r = 0.02 if i % 2 else -0.02
        own = (0.003 if i % 3 else -0.003) + (gap if i == n - 1 else 0.0)
        q.append(q[-1] * (1 + r))
        s.append(s[-1] * (1 + r + own))
    return Prices(
        {
            BENCHMARK: tuple(Bar(d, c, 1e6) for d, c in zip(days, q, strict=True)),
            "TRACK": tuple(Bar(d, c, 1e6) for d, c in zip(days, s, strict=True)),
        },
        (),
    )


def test_the_yardstick_is_the_usual_move_against_qqq_not_the_stocks_own_swings() -> None:
    """A stock that moves with QQQ: a 2% gap is small next to its own 2% swings
    but large next to its usual 0.3% gap from QQQ, so it passes."""
    z, _, _ = moves("TRACK", _tracking(0.02), DAY)
    assert z is not None and z >= CFG.move_sigma
    z_none, _, _ = moves("TRACK", _tracking(0.0), DAY)
    assert z_none is not None and abs(z_none) < CFG.move_sigma


def test_the_move_tested_is_not_part_of_its_own_yardstick() -> None:
    """With the move inside its yardstick, a 1-session z can never pass about
    4.25 however large the move; outside it, a huge move reads as huge."""
    z, window, _ = moves("BIG", _prices(BIG=0.5), DAY)
    assert window == 1 and z is not None and z > 20


def _random_walk(n: int, seed: int) -> tuple[list[float], list[float]]:
    """Irregular closes for a stock and QQQ, with the stock's own volatility
    ramping up, so a yardstick one return short or one session off differs."""
    rng = random.Random(seed)
    s, q = [100.0], [500.0]
    for i in range(1, n):
        m = rng.gauss(0, 0.01)
        q.append(q[-1] * (1 + m))
        s.append(s[-1] * (1 + m + rng.gauss(0, 0.002 + 0.0004 * i)))
    return s, q


def _expected_z(s: list[float], q: list[float], n: int) -> float:
    """The rule written out from return indices: the move over the last n
    sessions, over the 20 daily gaps from QQQ that end where the move begins."""
    last = len(s) - 1
    x = (s[last] / s[last - n] - 1) - (q[last] / q[last - n] - 1)
    gaps = [
        (s[t] / s[t - 1] - 1) - (q[t] / q[t - 1] - 1) for t in range(last - n - 19, last - n + 1)
    ]
    assert len(gaps) == 20
    return x / (statistics.stdev(gaps) * math.sqrt(n))


@pytest.mark.parametrize(("length", "seed"), [(26, 1), (27, 2), (40, 3), (60, 4)])
def test_the_move_matches_the_rule_written_out_by_hand(length: int, seed: int) -> None:
    s, q = _random_walk(length, seed)
    days = _days(length)
    prices = Prices(
        {
            BENCHMARK: tuple(Bar(d, c, 1e6) for d, c in zip(days, q, strict=True)),
            "W": tuple(Bar(d, c, 1e6) for d, c in zip(days, s, strict=True)),
        },
        (),
    )
    z, window, note = moves("W", prices, DAY)
    by_window = {n: _expected_z(s, q, n) for n in (1, 5)}
    best = max(by_window, key=lambda n: abs(by_window[n]))
    assert note is None and window == best
    assert z == pytest.approx(by_window[best])


@pytest.mark.parametrize(
    ("length", "window", "note"),
    [
        (21, None, "too little price history for the 1-session move"),
        (22, 1, "too little price history for the 5-session move"),
        (25, 1, "too little price history for the 5-session move"),
        (26, None, None),
    ],
)
def test_short_histories_say_which_move_could_not_be_tested(
    length: int, window: int | None, note: str | None
) -> None:
    prices = Prices({BENCHMARK: _flat(length), "NEW": _bars(0.0, length)}, ())
    z, got_window, got_note = moves("NEW", prices, DAY)
    if length == 21:
        assert z is None and got_note == (
            "too little price history for the 1-session move; "
            "too little price history for the 5-session move"
        )
    else:
        assert got_note == note
        if window is not None:
            assert got_window == window
    out = screen(["NEW"], DAY, DAY, CFG, prices, None)
    # Only a stock with both moves tested is a screen decision.
    assert (out["unscreened"] == ["NEW"]) is (length < 26)


def test_a_missing_bar_is_not_read_as_a_one_session_move() -> None:
    """Without the day before the session, the '1-session' move would span two
    sessions but be measured against a one-session yardstick."""
    bars = _bars(0.03)
    holed = bars[:-2] + bars[-1:]
    prices = Prices({BENCHMARK: _flat(), "HOLE": holed}, ())
    z, window, note = moves("HOLE", prices, DAY)
    assert z is None and window is None
    assert note == "gap in prices for the 1-session move; gap in prices for the 5-session move"
    old = bars[:-30] + bars[-29:]  # a hole long before the windows tested
    z, window, note = moves("OLD", Prices({BENCHMARK: _flat(), "OLD": old}, ()), DAY)
    assert note is None and window in (1, 5)


def test_a_stock_with_no_usual_move_says_so() -> None:
    prices = Prices({BENCHMARK: _flat(), "STILL": _flat()}, ())
    assert moves("STILL", prices, DAY) == (
        None,
        None,
        "no usual move for the 1-session move; no usual move for the 5-session move",
    )


# who passes


def test_a_new_filing_or_a_big_move_passes_and_nothing_else_does() -> None:
    filings = Filings(
        {
            "FILED": [_meta("FILED", "8-K", DAY)],
            "OLD": [_meta("OLD", "10-Q", date(2026, 11, 5))],  # on the previous session: old
            "OTHER": [_meta("OTHER", "S-1", DAY)],  # not a form the screen counts
        }
    )
    prices = _prices(FILED=0.0, OLD=0.0, OTHER=0.0, BIG=0.08, QUIET=0.002)
    cfg = replace(CFG, fill_to=0)
    out = screen(["FILED", "OLD", "OTHER", "BIG", "QUIET"], DAY, DAY, cfg, prices, filings)
    assert [r["ticker"] for r in out["chosen"]] == ["FILED", "BIG"]
    assert out["chosen"][0]["reasons"] == [f"8-K filed {DAY.isoformat()}"]
    assert out["chosen"][1]["reasons"][0].startswith("moved ")
    assert out["held"] == ["OLD", "OTHER", "QUIET"]
    assert out["filled"] == [] and out["over_cap"] == []
    assert out["filings_checked"] is True and out["since"] == "2026-11-05"


def test_a_big_down_move_passes_too() -> None:
    out = screen(["DOWN"], DAY, DAY, replace(CFG, fill_to=0), _prices(DOWN=-0.08), None)
    assert [r["ticker"] for r in out["chosen"]] == ["DOWN"]
    assert "below QQQ" in out["chosen"][0]["reasons"][0]


def test_on_monday_a_weekend_filing_counts_and_fridays_does_not() -> None:
    monday = date(2026, 11, 9)
    days = _days(40, monday)
    swing = [100.0 if i % 2 == 0 else 101.0 for i in range(40)]
    bars = {
        t: tuple(Bar(d, c, 1e6) for d, c in zip(days, swing, strict=True)) for t in ("SAT", "FRI")
    }
    bars[BENCHMARK] = tuple(Bar(d, 500.0, 1e6) for d in days)
    filings = Filings(
        {"SAT": [_meta("SAT", "8-K", date(2026, 11, 7))], "FRI": [_meta("FRI", "8-K", DAY)]}
    )
    out = screen(["SAT", "FRI"], monday, monday, replace(CFG, fill_to=0), Prices(bars, ()), filings)
    assert [r["ticker"] for r in out["chosen"]] == ["SAT"]
    assert out["since"] == "2026-11-06" and out["held"] == ["FRI"]


def test_past_the_cap_filings_come_first_then_the_bigger_move() -> None:
    filings = Filings({"FILED": [_meta("FILED", "10-K", DAY)]})
    prices = _prices(SMALL=0.06, HUGE=0.15, FILED=0.0)
    cfg = replace(CFG, cap=2, fill_to=0)
    out = screen(["SMALL", "HUGE", "FILED"], DAY, DAY, cfg, prices, filings)
    assert [r["ticker"] for r in out["chosen"]] == ["FILED", "HUGE"]
    assert [r["ticker"] for r in out["over_cap"]] == ["SMALL"]


def test_a_recent_call_is_no_reason_to_pass(tmp_path: Path) -> None:
    """Dropped 2026-11-08 (see config/screen.yaml): the screen reads no records."""
    out = screen(["QUIET"], DAY, DAY, replace(CFG, fill_to=0), _prices(QUIET=0.0), None)
    assert out["held"] == ["QUIET"]
    assert "recent" not in json.dumps(out)


def test_an_edgar_failure_for_one_stock_is_noted_and_the_rest_still_screen() -> None:
    filings = Filings({"FILED": [_meta("FILED", "8-K", DAY)]}, broken=frozenset({"DOWN"}))
    prices = _prices(FILED=0.0, DOWN=0.0)
    out = screen(["DOWN", "FILED"], DAY, DAY, replace(CFG, fill_to=0), prices, filings)
    assert [r["ticker"] for r in out["chosen"]] == ["FILED"]
    assert out["unchecked"] == {"DOWN": ["filings unavailable (ConnectionError)"]}
    assert out["unscreened"] == ["DOWN"] and out["held"] == []  # no decision, so not a Hold
    assert out["filings_failed"] == 1


def test_a_stock_that_could_not_be_checked_is_never_drawn_for_the_fill() -> None:
    filings = Filings({}, broken=frozenset({"DOWN"}))
    out = screen(["DOWN", "OK"], DAY, DAY, CFG, _prices(DOWN=0.0, OK=0.0), filings)
    assert out["filled"] == ["OK"] and out["unscreened"] == ["DOWN"]
    assert "Could not be screened (no decision): DOWN" in "\n".join(lines(out))


def test_without_filings_only_moves_count_and_the_output_says_so() -> None:
    out = screen(["BIG"], DAY, DAY, replace(CFG, fill_to=0), _prices(BIG=0.08), None)
    assert out["filings_checked"] is False
    assert "Filings were not checked" in "\n".join(lines(out))


# the random fill


def test_spare_slots_are_filled_at_random_from_the_stocks_left_out() -> None:
    quiet = [f"Q{i:02d}" for i in range(30)]
    prices = _prices(BIG=0.08, **dict.fromkeys(quiet, 0.0))
    out = screen(["BIG", *quiet], DAY, DAY, CFG, prices, None)
    assert [r["ticker"] for r in out["chosen"]] == ["BIG"]
    assert len(out["filled"]) == CFG.fill_to - 1
    assert set(out["filled"]) <= set(quiet)
    assert set(out["filled"]).isdisjoint(out["held"])
    assert len(out["held"]) == 30 - len(out["filled"])
    text = "\n".join(lines(out))
    assert "19 more drawn at random" in text
    assert "Random fill (a check on the screen): " in text


def test_no_fill_when_the_screen_already_reaches_fill_to() -> None:
    movers = {f"M{i:02d}": 0.08 for i in range(3)}
    prices = _prices(QUIET=0.0, **movers)
    out = screen([*movers, "QUIET"], DAY, DAY, replace(CFG, fill_to=3), prices, None)
    assert len(out["chosen"]) == 3 and out["filled"] == [] and out["held"] == ["QUIET"]


def test_the_fill_is_the_same_on_a_rerun_and_changes_by_day() -> None:
    left = [f"S{i:02d}" for i in range(50)]
    first = random_fill(left, 9, "screen-v2", DAY)
    assert first == random_fill(list(reversed(left)), 9, "screen-v2", DAY)
    assert first != random_fill(left, 9, "screen-v2", date(2026, 11, 5))
    assert len(set(first)) == 9


@pytest.mark.parametrize(
    ("left", "slots", "count"), [([], 5, 0), (["A"], 0, 0), (["A", "B"], 5, 2)]
)
def test_the_fill_never_draws_more_than_it_can(left: list[str], slots: int, count: int) -> None:
    assert len(random_fill(left, slots, "screen-v2", DAY)) == count


# the command


NOW = datetime(2026, 11, 7, 15, tzinfo=UTC)  # Saturday: the 11-06 session is the latest


def test_main_prints_the_screen_and_writes_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def fetch(tickers: object, source: object, *, now: datetime) -> Prices:
        return _prices(**dict.fromkeys(load_universe(), 0.0) | {"CRWD": 0.08})

    filings = Filings({"DDOG": [_meta("DDOG", "8-K", date(2026, 11, 6))]})
    out = tmp_path / "screen.json"
    assert main(["--json", str(out)], prices_fetch=fetch, filings=filings, now=NOW) == 0
    text = capsys.readouterr().out
    assert "Screen screen-v2 for the 2026-11-06 session" in text
    result = json.loads(out.read_text())
    assert {r["ticker"] for r in result["chosen"]} == {"CRWD", "DDOG"}
    assert len(result["filled"]) == CFG.fill_to - 2
    assert filings.asked == 113  # every stock in the universe is looked up once


def test_main_refuses_a_qqq_bar_that_is_a_session_behind(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fetch(tickers: object, source: object, *, now: datetime) -> Prices:
        return _prices(CRWD=0.0)  # QQQ's last bar is 2026-11-06

    monday_evening = datetime(2026, 11, 9, 23, tzinfo=UTC)
    assert main(["--no-filings"], prices_fetch=fetch, now=monday_evening) == 1
    assert "no QQQ bar for the 2026-11-09 session yet" in capsys.readouterr().out


def _all_priced(tickers: object, source: object, *, now: datetime) -> Prices:
    return _prices(**dict.fromkeys(load_universe(), 0.0))


def test_main_fails_when_too_many_edgar_lookups_fail(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "screen.json"
    filings = Filings({}, broken=frozenset(load_universe()[:12]))
    assert main(["--json", str(out)], prices_fetch=_all_priced, filings=filings, now=NOW) == 1
    assert "12 EDGAR lookups failed" in capsys.readouterr().out
    refused = json.loads(out.read_text())
    assert refused["ok"] is False and refused["refused"] == ["12 EDGAR lookups failed"]
    few = Filings({}, broken=frozenset(load_universe()[:11]))
    assert main(["--json", str(out)], prices_fetch=_all_priced, filings=few, now=NOW) == 0
    kept = json.loads(out.read_text())
    assert kept["ok"] is True and kept["refused"] == []


@pytest.mark.parametrize(("missing", "code"), [(12, 1), (11, 0)])
def test_main_fails_when_too_many_stocks_have_no_prices(
    missing: int, code: int, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Yahoo can serve QQQ and drop many stocks: those would all be
    'could not be screened', and the run must not call that a screen."""
    gone = set(load_universe()[:missing])

    def fetch(tickers: object, source: object, *, now: datetime) -> Prices:
        return _prices(**{t: 0.0 for t in load_universe() if t not in gone})

    out = tmp_path / "screen.json"
    assert main(["--no-filings", "--json", str(out)], prices_fetch=fetch, now=NOW) == code
    result = json.loads(out.read_text())
    assert result["prices_failed"] == missing and result["ok"] is (code == 0)
    if code:
        assert f"{missing} stocks' prices could not be fully checked" in capsys.readouterr().out


def test_main_without_qqq_screens_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    def fetch(tickers: object, source: object, *, now: datetime) -> Prices:
        return Prices({}, ())

    assert main(["--no-filings"], prices_fetch=fetch, now=NOW) == 1
    assert "no QQQ prices" in capsys.readouterr().out
