"""The daily screen (screen-v7): a stock passes when its earnings are due soon."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from conftest import NoNetwork

from tenq.config import load_watchlist
from tenq.earnings import RETRY_PAUSE_S, ROWS, YahooEarnings, lookup
from tenq.prices import BENCHMARK, MARKET_TZ, Bar, Prices
from tenq.screen import (
    just_reported,
    lines,
    load_config,
    load_universe,
    main,
    next_release,
    picked,
    random_fill,
    refusals,
    run_screen,
    screen,
    universe_names,
    window,
)

DAY = date(2026, 11, 6)  # a Friday session: the 5 sessions after it run 11-09 to 11-13
CFG = load_config()
# Any edit to screen.yaml, even a comment, needs a new version; pin each one.
KNOWN_SCREENS = {
    "screen-v6": "0b529aacebbb522a78f00cffcb1911d79f86600802788f9499d0b7623eb6ab5d",
    "screen-v7": "05c2566b6a4e48d2a9a83c0318a774880cc7f6d68ab11f9a3174eea0de056a9a",
}
LATER = date(2026, 12, 15)  # a release well past the window
LATER_RELEASE = datetime(2026, 12, 15, 16, tzinfo=MARKET_TZ)


def _at(d: date, hour: int = 16) -> datetime:
    return datetime.combine(d, time(hour), tzinfo=MARKET_TZ)


RUN = datetime(2026, 11, 6, 22, 30, tzinfo=UTC)  # the evening run: 17:30 in New York


class Dates:
    """A stand-in for Yahoo's earnings list."""

    def __init__(
        self,
        table: dict[str, list[datetime]] | None = None,
        *,
        broken: frozenset[str] = frozenset(),
        flaky: frozenset[str] = frozenset(),
    ) -> None:
        self.table = table or {}
        self.broken, self.flaky = broken, set(flaky)
        self.asked: list[str] = []

    def dates(self, ticker: str) -> list[datetime]:
        self.asked.append(ticker)
        if ticker in self.broken:
            raise ConnectionError("Yahoo down")
        if ticker in self.flaky:
            self.flaky.discard(ticker)  # fails once, then answers
            raise ConnectionError("throttled")
        return self.table.get(ticker, [_at(LATER)])


def _no_sleep(seconds: float) -> None:
    pass


def _screen(
    table: dict[str, list[datetime]],
    tickers: list[str] | None = None,
    cfg: Any = CFG,
    **kw: Any,
) -> dict[str, Any]:
    now = kw.pop("now", RUN)
    return screen(tickers or list(table), DAY, cfg, Dates(table, **kw), now=now, sleep=_no_sleep)


def _tickers(rows: list[dict[str, Any]]) -> list[str]:
    return [r["ticker"] for r in rows]


# settings


def test_the_settings_load() -> None:
    assert CFG.version == "screen-v7"
    assert (CFG.earnings_sessions, CFG.cap, CFG.fill_to) == (5, 25, 20)
    assert CFG.sha256 == KNOWN_SCREENS[CFG.version], "screen.yaml changed: bump the version"


@pytest.mark.parametrize(
    ("edit", "says"),
    [
        (lambda t: t.replace("cap: 25", "cap: 25\nmove_sigma: 2.5"), "expected exactly"),
        (lambda t: t.replace("earnings_sessions: 5\n", ""), "expected exactly"),
        (lambda t: t.replace("earnings_sessions: 5", "earnings_sessions: 0"), "earnings_sessions"),
        (lambda t: t.replace("earnings_sessions: 5", "earnings_sessions: 21"), "earnings_sessions"),
        (lambda t: t.replace("earnings_sessions: 5", "earnings_sessions: true"), "earnings_"),
        (lambda t: t.replace("earnings_sessions: 5", "earnings_sessions: 2.5"), "earnings_"),
        (lambda t: t.replace("fill_to: 20", "fill_to: 30"), "fill_to"),
        (lambda t: t.replace("fill_to: 20", "fill_to: -1"), "fill_to"),
        (lambda t: t.replace("cap: 25", "cap: true"), "cap"),
        (lambda t: t.replace("cap: 25", "cap: 0"), "cap"),
        (lambda t: t.replace("version: screen-v7", "version: ''"), "version"),
    ],
)
def test_bad_settings_are_refused(tmp_path: Path, edit: object, says: str) -> None:
    text = (Path(__file__).parents[1] / "config" / "screen.yaml").read_text()
    path = tmp_path / "screen.yaml"
    path.write_text(edit(text))  # type: ignore[operator]
    with pytest.raises(ValueError, match=says):
        load_config(path)


def test_the_window_is_read_from_the_file(tmp_path: Path) -> None:
    text = (Path(__file__).parents[1] / "config" / "screen.yaml").read_text()
    path = tmp_path / "screen.yaml"
    path.write_text(text.replace("earnings_sessions: 5", "earnings_sessions: 3"))
    cfg = load_config(path)
    assert cfg.earnings_sessions == 3
    assert _screen({"WED": [_at(date(2026, 11, 11))], "THU": [_at(date(2026, 11, 12))]},
                   cfg=replace(cfg, fill_to=0))["held"] == ["THU"]  # fmt: skip


def test_the_universe_is_the_watchlist_first_then_the_nasdaq_100_once_each() -> None:
    tickers = load_universe()
    watch = load_watchlist().tickers
    assert tickers[: len(watch)] == tuple(watch)
    assert len(tickers) == len(set(tickers)) == 113
    assert "GOOGL" in tickers and "GOOG" not in tickers


# the window and the release date


@pytest.mark.parametrize(
    ("day", "now", "sessions", "first", "last"),
    [
        (DAY, RUN, 5, date(2026, 11, 9), date(2026, 11, 13)),  # over the weekend
        (DAY, RUN, 1, date(2026, 11, 9), date(2026, 11, 9)),
        (
            date(2026, 11, 20),
            datetime(2026, 11, 20, 22, 30, tzinfo=UTC),
            5,
            date(2026, 11, 23),
            date(2026, 11, 30),
        ),  # Thanksgiving
        (DAY, _at(date(2026, 11, 9), hour=7), 5, date(2026, 11, 9), date(2026, 11, 13)),
        # never before the session after the screened one, however early the clock
        (DAY, _at(DAY, hour=7), 5, date(2026, 11, 9), date(2026, 11, 13)),
        # late enough to miss Monday's open: the calls enter Tuesday
        (DAY, _at(date(2026, 11, 9), hour=10), 5, date(2026, 11, 10), date(2026, 11, 16)),
    ],
)
def test_the_window_is_the_sessions_from_the_first_open_after_the_run(
    day: date, now: datetime, sessions: int, first: date, last: date
) -> None:
    assert window(day, now, sessions) == (first, last)


def test_the_next_release_is_the_first_still_to_come_at_the_run() -> None:
    past, today, soon = _at(date(2026, 11, 5)), _at(DAY), _at(date(2026, 11, 10))
    assert next_release([soon, past, today], RUN) == date(2026, 11, 10)
    assert next_release([past, today], RUN) is None
    assert next_release([], RUN) is None
    assert next_release([_at(DAY, hour=18)], RUN) == DAY  # after the run: still to come


def test_a_release_is_dated_in_new_york_not_utc() -> None:
    evening = datetime(2026, 11, 14, 3, tzinfo=UTC)  # 22:00 on 11-13 in New York
    assert next_release([evening], RUN) == date(2026, 11, 13)
    morning = datetime(2026, 11, 9, 12, tzinfo=UTC)  # 07:00 on 11-09 in New York
    assert next_release([morning], RUN) == date(2026, 11, 9)
    out = _screen({"LATE": [evening]}, cfg=replace(CFG, fill_to=0))
    assert _tickers(out["chosen"]) == ["LATE"]  # due 11-13 in New York, the 5th session


def test_a_late_run_counts_from_the_open_it_can_still_make() -> None:
    late = _at(date(2026, 11, 9), hour=10)  # Monday 10:00: the 11-06 session, entry Tuesday
    out = _screen(
        {
            "OUT": [_at(date(2026, 11, 9), hour=7), LATER_RELEASE],  # out before the run
            "SIXTH": [_at(date(2026, 11, 16))],  # 5th session from Tuesday
            "SEVENTH": [_at(date(2026, 11, 17))],
        },
        cfg=replace(CFG, fill_to=0),
        now=late,
    )
    assert _tickers(out["chosen"]) == ["SIXTH"]
    assert out["held"] == ["OUT", "SEVENTH"]
    assert (out["due_from"], out["due_through"]) == ("2026-11-10", "2026-11-16")


# who passes


def test_a_release_due_in_the_window_passes_and_nothing_else_does() -> None:
    out = _screen(
        {
            "SOON": [_at(date(2026, 11, 13))],  # the 5th session after
            "LATER": [_at(date(2026, 11, 16))],  # the 6th
            "TODAY": [_at(DAY), _at(date(2027, 2, 5))],  # out before the calls are made
        },
        cfg=replace(CFG, fill_to=0),
    )
    assert out["chosen"] == [
        {"ticker": "SOON", "earnings": "2026-11-13", "due": True, "unchecked": [],
         "reasons": ["earnings due 2026-11-13"]},
    ]  # fmt: skip
    assert out["held"] == ["LATER", "TODAY"]
    assert out["next_earnings"] == {"SOON": "2026-11-13", "LATER": "2026-11-16",
                                    "TODAY": "2027-02-05"}  # fmt: skip
    assert (out["due_from"], out["due_through"]) == ("2026-11-09", "2026-11-13")


def test_a_release_before_the_next_open_passes() -> None:
    out = _screen({"EARLY": [_at(date(2026, 11, 9), hour=7)]}, cfg=replace(CFG, fill_to=0))
    assert _tickers(out["chosen"]) == ["EARLY"]


SPREAD = {
    "A": date(2026, 11, 12),
    "B": date(2026, 11, 9),
    "C": date(2026, 11, 10),
    "D": date(2026, 11, 10),
    "E": date(2026, 11, 10),
    "F": date(2026, 11, 11),
}


def test_past_the_cap_the_soonest_release_comes_first() -> None:
    cfg = replace(CFG, cap=3, fill_to=0)
    out = _screen({t: [_at(d)] for t, d in SPREAD.items()}, cfg=cfg)
    chosen, over = _tickers(out["chosen"]), _tickers(out["over_cap"])
    assert chosen[0] == "B" and set(chosen[1:]) < {"C", "D", "E"}
    assert len(over) == 3 and over[0] in {"C", "D", "E"} and over[1:] == ["F", "A"]
    again = _screen({t: [_at(d)] for t, d in reversed(SPREAD.items())}, cfg=cfg)
    assert _tickers(again["chosen"]) == chosen  # the same on a rerun, in any order


def test_stocks_reporting_the_same_day_are_taken_in_a_drawn_order() -> None:
    table = {t: [_at(d)] for t, d in SPREAD.items()}
    orders = set()
    for i in range(20):
        cfg = replace(CFG, cap=2, fill_to=0, version=f"v{i}")
        orders.add(tuple(_tickers(_screen(table, cfg=cfg)["chosen"])))
    assert {o[0] for o in orders} == {"B"}
    assert len(orders) == 3  # each of C, D and E gets the second place sometimes


def test_a_recent_call_is_no_reason_to_pass() -> None:
    """Dropped 2026-11-08 (see config/screen.yaml): the screen reads no records."""
    out = _screen({"QUIET": [_at(LATER)]}, cfg=replace(CFG, fill_to=0))
    assert out["held"] == ["QUIET"]
    assert "recent" not in json.dumps(out)


# stocks that could not be screened


def test_a_failed_lookup_is_noted_and_the_rest_still_screen() -> None:
    out = _screen({"SOON": [_at(date(2026, 11, 10))]}, ["DOWN", "SOON"],
                  cfg=replace(CFG, fill_to=0), broken=frozenset({"DOWN"}))  # fmt: skip
    assert _tickers(out["chosen"]) == ["SOON"]
    assert out["unchecked"] == {"DOWN": ["earnings dates unavailable (ConnectionError)"]}
    assert out["unscreened"] == ["DOWN"] and out["held"] == []  # no decision, so not a Hold
    assert out["dates_failed"] == 1 and "DOWN" not in out["next_earnings"]


def test_a_stock_with_no_upcoming_date_is_not_screened() -> None:
    out = _screen({"GONE": [_at(date(2026, 8, 1))], "NONE": []}, cfg=replace(CFG, fill_to=0))
    assert out["unscreened"] == ["GONE", "NONE"] and out["held"] == []
    assert out["unchecked"]["NONE"] == ["no upcoming earnings date"]
    assert out["dates_failed"] == 2


def test_a_stock_that_just_reported_with_no_next_date_yet_is_held() -> None:
    out = _screen(
        {
            "TODAY": [_at(DAY, hour=7)],  # reported this morning; Yahoo has no next date yet
            "MONTH": [_at(date(2026, 10, 7))],  # 30 days before the run: still held
            "OLDER": [_at(date(2026, 10, 6))],  # 31 days: no decision
            "NEXT": [_at(DAY, hour=7), _at(LATER)],  # a next date: the usual way
        },
        cfg=replace(CFG, fill_to=0),
    )
    assert out["held"] == ["TODAY", "MONTH", "NEXT"] and out["unscreened"] == ["OLDER"]
    assert out["just_reported"] == {"TODAY": "2026-11-06", "MONTH": "2026-10-07"}
    assert out["next_earnings"] == {"NEXT": LATER.isoformat()}
    assert out["dates_failed"] == 1 and not refusals(out | {"universe": 20})


def test_a_stock_that_just_reported_can_be_drawn_for_the_fill() -> None:
    out = _screen({"TODAY": [_at(DAY, hour=7)]}, ["TODAY"])
    assert out["filled"] == ["TODAY"] and out["unscreened"] == []


def test_a_release_after_the_run_is_not_one_just_reported() -> None:
    assert just_reported([_at(DAY, hour=18)], RUN) is None  # still to come
    assert just_reported([_at(DAY, hour=16), _at(date(2026, 8, 6))], RUN) == DAY
    assert just_reported([], RUN) is None


def test_just_reported_counts_days_in_new_york_not_utc() -> None:
    late = datetime(2026, 11, 7, 3, tzinfo=UTC)  # 22:00 on the 6th in New York
    assert just_reported([_at(date(2026, 10, 7), hour=7)], late) == date(2026, 10, 7)
    evening = _at(date(2026, 10, 7), hour=21).astimezone(UTC)  # already the 8th in UTC
    assert just_reported([evening], RUN) == date(2026, 10, 7)


def test_a_stock_that_could_not_be_screened_is_never_drawn_for_the_fill() -> None:
    out = _screen({}, ["DOWN", "OK"], broken=frozenset({"DOWN"}))
    assert out["filled"] == ["OK"] and out["unscreened"] == ["DOWN"]
    text = "\n".join(lines(out))
    assert "Could not be screened (no decision):\n  DOWN: earnings dates unavailable" in text


def test_a_lookup_that_fails_is_asked_once_more_after_a_pause() -> None:
    pauses: list[float] = []
    source = Dates({"OK": [_at(LATER)]}, flaky=frozenset({"SLOW"}))
    found, failed = lookup(["SLOW", "OK"], source, sleep=pauses.append)
    assert set(found) == {"SLOW", "OK"} and failed == {}
    assert source.asked == ["SLOW", "OK", "SLOW"] and pauses == [RETRY_PAUSE_S]


def test_a_lookup_that_fails_twice_is_reported_not_raised() -> None:
    pauses: list[float] = []
    source = Dates(broken=frozenset({"DOWN"}))
    found, failed = lookup(["DOWN", "OK"], source, sleep=pauses.append)
    assert list(found) == ["OK"] and failed == {
        "DOWN": "earnings dates unavailable (ConnectionError)"
    }
    assert source.asked == ["DOWN", "OK", "DOWN"] and pauses == [RETRY_PAUSE_S]


def test_no_pause_when_every_lookup_answers() -> None:
    pauses: list[float] = []
    lookup(["A", "B"], Dates(), sleep=pauses.append)
    assert pauses == []


# the random fill


def test_spare_slots_are_filled_at_random_from_the_stocks_left_out() -> None:
    quiet = [f"Q{i:02d}" for i in range(30)]
    out = _screen({"SOON": [_at(date(2026, 11, 10))]}, ["SOON", *quiet])
    assert _tickers(out["chosen"]) == ["SOON"]
    assert len(out["filled"]) == CFG.fill_to - 1
    assert set(out["filled"]) <= set(quiet)
    assert set(out["filled"]).isdisjoint(out["held"])
    assert len(out["held"]) == 30 - len(out["filled"])
    text = "\n".join(lines(out))
    assert "1 of 31 stocks pass; 19 more drawn at random; 11 are Hold by screen." in text
    assert "Random fill (a check on the screen): " in text


def test_no_fill_when_the_screen_already_reaches_fill_to() -> None:
    due = {f"D{i}": [_at(date(2026, 11, 10))] for i in range(3)}
    out = _screen(due, [*due, "QUIET"], cfg=replace(CFG, fill_to=3))
    assert len(out["chosen"]) == 3 and out["filled"] == [] and out["held"] == ["QUIET"]


def test_no_fill_past_the_cap() -> None:
    due = {f"D{i}": [_at(date(2026, 11, 10))] for i in range(6)}
    out = _screen(due, [*due, "QUIET"], cfg=replace(CFG, cap=3, fill_to=3))
    assert len(out["chosen"]) == 3 and len(out["over_cap"]) == 3
    assert out["filled"] == [] and out["held"] == ["QUIET"]


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


def test_the_output_says_the_window_and_why_each_stock_passed() -> None:
    out = _screen({"SOON": [_at(date(2026, 11, 13))]}, cfg=replace(CFG, fill_to=0))
    text = lines(out)
    assert text[0] == ("Screen screen-v7 for the 2026-11-06 session (earnings due by 2026-11-13):")
    assert "   1. SOON: earnings due 2026-11-13" in text


# Yahoo's earnings list


REAL_DATES = YahooEarnings.dates  # read before the guard in conftest replaces it


@pytest.fixture
def parse_yahoo(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real reading of Yahoo's table, on a stand-in for yfinance's Ticker."""
    monkeypatch.setattr(YahooEarnings, "dates", REAL_DATES)


class _Ticker:
    def __init__(self, frame: pd.DataFrame | None) -> None:
        self.frame = frame
        self.limits: list[int] = []

    def get_earnings_dates(self, limit: int) -> pd.DataFrame | None:
        self.limits.append(limit)
        return self.frame


@pytest.mark.usefixtures("parse_yahoo")
def test_yahoos_release_times_are_kept_in_new_york_time(monkeypatch: pytest.MonkeyPatch) -> None:
    index = pd.DatetimeIndex(["2026-12-02 16:00", "2026-09-02 16:00"]).tz_localize(MARKET_TZ)
    fake = _Ticker(pd.DataFrame({"EPS Estimate": [1.0, 2.0]}, index=index))
    monkeypatch.setattr("yfinance.Ticker", lambda t: fake)
    got = YahooEarnings().dates("CRWD")
    assert got == [_at(date(2026, 12, 2)), _at(date(2026, 9, 2))]
    assert fake.limits == [ROWS]


@pytest.mark.usefixtures("parse_yahoo")
def test_yahoos_times_with_no_zone_are_read_as_new_york(monkeypatch: pytest.MonkeyPatch) -> None:
    index = pd.DatetimeIndex(["2026-12-02 07:00"])
    monkeypatch.setattr("yfinance.Ticker", lambda t: _Ticker(pd.DataFrame({"x": [1]}, index=index)))
    assert YahooEarnings().dates("CRWD") == [_at(date(2026, 12, 2), hour=7)]


@pytest.mark.usefixtures("parse_yahoo")
def test_no_list_from_yahoo_is_a_failed_lookup_and_asked_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # yfinance returns no table for a blocked or broken page: not "no date"
    monkeypatch.setattr("yfinance.Ticker", lambda t: _Ticker(None))
    with pytest.raises(LookupError):
        YahooEarnings().dates("CRWD")
    pauses: list[float] = []
    found, failed = lookup(["CRWD"], YahooEarnings(), sleep=pauses.append)
    assert (found, failed, pauses) == (
        {}, {"CRWD": "earnings dates unavailable (LookupError)"}, [RETRY_PAUSE_S]
    )  # fmt: skip


def test_the_earnings_lookup_never_reaches_yahoo_in_tests() -> None:
    with pytest.raises(NoNetwork):
        YahooEarnings().dates("CRWD")


# the whole run

NOW = datetime(2026, 11, 7, 15, tzinfo=UTC)  # Saturday: the 11-06 session is the latest


def _qqq(last: date = DAY) -> tuple[Bar, ...]:
    return tuple(Bar(last - timedelta(days=i), 500.0, 1e6) for i in range(5, -1, -1))


def _qqq_only(tickers: Sequence[str], source: object, *, now: datetime) -> Prices:
    assert tuple(tickers) == ()  # QQQ alone: fetch adds it itself
    return Prices({BENCHMARK: _qqq()}, ())


def test_main_prints_the_screen_and_writes_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dates = Dates({"DDOG": [_at(date(2026, 11, 12))], "CRWD": [_at(date(2026, 11, 10))]})
    out = tmp_path / "screen.json"
    code = main(["--json", str(out)], prices_fetch=_qqq_only, earnings=dates, now=NOW,
                sleep=_no_sleep)  # fmt: skip
    assert code == 0
    text = capsys.readouterr().out
    assert "Screen screen-v7 for the 2026-11-06 session" in text
    result = json.loads(out.read_text())
    assert _tickers(result["chosen"]) == ["CRWD", "DDOG"]  # the sooner release first
    assert len(result["filled"]) == CFG.fill_to - 2
    assert sorted(dates.asked) == sorted(load_universe())  # every stock looked up once


def test_main_refuses_a_qqq_bar_that_is_a_session_behind(
    capsys: pytest.CaptureFixture[str],
) -> None:
    monday_evening = datetime(2026, 11, 9, 23, tzinfo=UTC)
    assert main([], prices_fetch=_qqq_only, earnings=Dates(), now=monday_evening) == 1
    assert "no QQQ bar for the 2026-11-09 session yet" in capsys.readouterr().out


@pytest.mark.parametrize(("failing", "code"), [(12, 1), (11, 0)])
def test_main_fails_when_too_many_stocks_have_no_usable_date(
    failing: int, code: int, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Yahoo can answer for some stocks and not others: those would all be
    'could not be screened', and the run must not call that a screen."""
    gone = load_universe()[:failing]
    broken = frozenset(gone[::2])
    empty: dict[str, list[datetime]] = {t: [] for t in gone[1::2]}  # both kinds count
    out = tmp_path / "screen.json"
    dates = Dates(empty, broken=broken)
    assert main(["--json", str(out)], prices_fetch=_qqq_only, earnings=dates, now=NOW,
                sleep=_no_sleep) == code  # fmt: skip
    result = json.loads(out.read_text())
    assert result["dates_failed"] == failing and result["ok"] is (code == 0)
    if code:
        assert result["refused"] == [f"{failing} stocks' earnings dates could not be read"]
        assert "could not be read: too many to trust this screen" in capsys.readouterr().out


def test_main_without_qqq_screens_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    def fetch(tickers: object, source: object, *, now: datetime) -> Prices:
        return Prices({}, ())

    assert main([], prices_fetch=fetch, earnings=Dates(), now=NOW) == 1
    assert "no QQQ prices" in capsys.readouterr().out


def test_run_screen_never_raises_and_says_why_it_failed(caplog: pytest.LogCaptureFixture) -> None:
    def broken(tickers: object, source: object, *, now: datetime) -> Prices:
        raise OSError("disk")

    def no_qqq(tickers: object, source: object, *, now: datetime) -> Prices:
        return Prices({}, ())

    assert run_screen(NOW, prices_fetch=broken, earnings=Dates()) == {
        "kind": "screen",
        "ok": False,
        "refused": ["the screen failed (OSError)"],
    }
    assert "OSError: disk" in caplog.text  # the traceback is logged, the email gets the type only
    assert run_screen(NOW, prices_fetch=no_qqq, earnings=Dates())["refused"] == ["no QQQ prices"]
    late = run_screen(datetime(2026, 11, 9, 23, tzinfo=UTC), prices_fetch=_qqq_only,
                      earnings=Dates())  # fmt: skip
    assert late["ok"] is False and "2026-11-09 session" in late["refused"][0]


def test_run_screen_marks_a_result_too_broken_to_trust() -> None:
    dates = Dates(broken=frozenset(load_universe()[:12]))
    result = run_screen(NOW, prices_fetch=_qqq_only, earnings=dates, sleep=_no_sleep)
    assert result["ok"] is False
    assert result["refused"] == ["12 stocks' earnings dates could not be read"]
    assert "chosen" in result  # the detail is kept for the record


def test_the_models_get_the_stocks_that_passed_then_the_random_fill() -> None:
    dates = Dates({"DDOG": [_at(date(2026, 11, 12))], "CRWD": [_at(date(2026, 11, 10))]})
    result = run_screen(NOW, prices_fetch=_qqq_only, earnings=dates, sleep=_no_sleep)
    assert result["ok"] is True and result["day"] == "2026-11-06"
    got = picked(result)
    assert got[:2] == ["CRWD", "DDOG"]
    assert got[2:] == result["filled"] and len(got) == CFG.fill_to


def test_every_stock_the_screen_can_pick_has_a_name() -> None:
    names = universe_names()
    assert set(load_universe()) <= set(names)
    assert names["CRWD"] == load_watchlist().names["CRWD"]


@pytest.mark.parametrize(("failed", "refused"), [(2, False), (3, True)])
def test_exactly_one_stock_in_ten_without_a_date_is_still_trusted(
    failed: int, refused: bool
) -> None:
    assert bool(refusals({"dates_failed": failed, "universe": 20})) is refused
