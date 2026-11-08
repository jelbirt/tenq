"""The daily screen: which stocks go to Jev and Opus, decided by code alone."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from tenq.config import load_watchlist
from tenq.filings import FilingMeta
from tenq.prices import BENCHMARK, Bar, Prices, price_facts
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
    assert CFG.version == "screen-v1"
    assert (CFG.move_sigma, CFG.cap, CFG.fill_to) == (2.5, 25, 20)
    assert CFG.forms == ("8-K", "10-Q", "10-K")
    assert len(CFG.sha256) == 64


@pytest.mark.parametrize(
    ("edit", "says"),
    [
        (lambda t: t.replace("cap: 25", "cap: 25\nrecent_sessions: 20"), "expected exactly"),
        (lambda t: t.replace("fill_to: 20", "fill_to: 30"), "fill_to"),
        (lambda t: t.replace("fill_to: 20", "fill_to: -1"), "fill_to"),
        (lambda t: t.replace("cap: 25", "cap: true"), "cap"),
        (lambda t: t.replace("move_sigma: 2.5", "move_sigma: 0"), "move_sigma"),
        (lambda t: t.replace("forms: [8-K, 10-Q, 10-K]", "forms: []"), "forms"),
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
    facts = price_facts("STEADY", prices)
    assert facts is not None and facts.volatility is not None
    x = facts.excess[5]
    assert x is not None
    assert window == 5
    assert z == pytest.approx(x / (facts.volatility / math.sqrt(252) * math.sqrt(5)))
    assert z is not None and z >= CFG.move_sigma


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
    first = random_fill(left, 9, "screen-v1", DAY)
    assert first == random_fill(list(reversed(left)), 9, "screen-v1", DAY)
    assert first != random_fill(left, 9, "screen-v1", date(2026, 11, 5))
    assert len(set(first)) == 9


@pytest.mark.parametrize(
    ("left", "slots", "count"), [([], 5, 0), (["A"], 0, 0), (["A", "B"], 5, 2)]
)
def test_the_fill_never_draws_more_than_it_can(left: list[str], slots: int, count: int) -> None:
    assert len(random_fill(left, slots, "screen-v1", DAY)) == count


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
    assert "Screen screen-v1 for the 2026-11-06 session" in text
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


def test_main_fails_when_too_many_edgar_lookups_fail(capsys: pytest.CaptureFixture[str]) -> None:
    def fetch(tickers: object, source: object, *, now: datetime) -> Prices:
        return _prices(CRWD=0.0)

    filings = Filings({}, broken=frozenset(load_universe()[:12]))
    assert main([], prices_fetch=fetch, filings=filings, now=NOW) == 1
    assert "12 EDGAR lookups failed" in capsys.readouterr().out
    few = Filings({}, broken=frozenset(load_universe()[:11]))
    assert main([], prices_fetch=fetch, filings=few, now=NOW) == 0


def test_main_without_qqq_screens_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    def fetch(tickers: object, source: object, *, now: datetime) -> Prices:
        return Prices({}, ())

    assert main(["--no-filings"], prices_fetch=fetch, now=NOW) == 1
    assert "no QQQ prices" in capsys.readouterr().out
