from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
import pytest

from tenq import prices
from tenq.prices import (
    BENCHMARK,
    Bar,
    Prices,
    aligned,
    bars_from_frame,
    drop_unfinished,
    fetch,
    price_facts,
    trailing_return,
    volatility,
)

DAYS = pd.bdate_range("2026-09-01", periods=30)


def _frame(series: dict[str, Sequence[float]], days: Any = DAYS) -> pd.DataFrame:
    """A frame shaped like yfinance's batch download: (field, ticker) columns."""
    cols = {}
    for k, (ticker, closes) in enumerate(series.items()):
        cols[("Close", ticker)] = list(closes)
        cols[("Volume", ticker)] = [1000.0 * (k + 1)] * len(closes)
    frame = pd.DataFrame(cols, index=pd.DatetimeIndex(days[: len(next(iter(series.values())))]))
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    return frame


class Source:
    def __init__(self, *frames: Any) -> None:
        self.frames = list(frames)
        self.calls: list[tuple[list[str], str]] = []

    def history(self, tickers: Sequence[str], period: str) -> Any:
        self.calls.append((list(tickers), period))
        out = self.frames.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


AFTER_CLOSE = datetime(2026, 10, 13, 23, 0, tzinfo=UTC)  # 19:00 New York, a later day


def _no_sleep(_: float) -> None:
    return None


def test_bars_are_read_per_ticker_and_bad_closes_skipped() -> None:
    frame = _frame({"CRWD": [100.0, math.nan, 0.0, -1.0, 104.0], BENCHMARK: [10.0] * 5})
    bars = bars_from_frame(frame, ["CRWD", BENCHMARK, "MISSING"])
    assert [b.close for b in bars["CRWD"]] == [100.0, 104.0]
    assert bars["CRWD"][0].day == date(2026, 9, 1)
    assert bars["CRWD"][0].volume == 1000.0
    assert bars[BENCHMARK][0].volume == 2000.0
    assert "MISSING" not in bars


def test_one_ticker_with_flat_columns_is_read() -> None:
    frame = pd.DataFrame({"Close": [5.0, 6.0], "Volume": [1.0, 2.0]}, index=DAYS[:2])
    assert [b.close for b in bars_from_frame(frame, ["QQQ"])["QQQ"]] == [5.0, 6.0]
    assert bars_from_frame(frame, ["QQQ", "CRWD"]) == {}


def test_an_all_nan_ticker_is_left_out_with_a_note() -> None:
    nan = _frame({"GONE": [math.nan] * 2})
    source = Source(
        _frame({"CRWD": [1.0, 2.0], "GONE": [math.nan] * 2, BENCHMARK: [1.0, 1.0]}), nan, nan
    )
    out = fetch(["CRWD", "GONE"], source, now=AFTER_CLOSE, sleep=_no_sleep)
    assert set(out.bars) == {"CRWD", BENCHMARK}
    assert out.notes == ("no prices for GONE",)
    assert [c[0] for c in source.calls] == [["CRWD", "GONE", BENCHMARK], ["GONE"], ["GONE"]]


def test_tickers_missing_from_a_batch_are_asked_for_again() -> None:
    slept: list[float] = []
    source = Source(
        _frame({"CRWD": [1.0, 2.0], "DDOG": [math.nan] * 2, BENCHMARK: [1.0, 1.0]}),
        _frame({"DDOG": [3.0, 4.0]}),
    )
    out = fetch(["CRWD", "DDOG"], source, now=AFTER_CLOSE, sleep=slept.append)
    assert [b.close for b in out.bars["DDOG"]] == [3.0, 4.0]
    assert out.notes == ()
    assert [c[0] for c in source.calls] == [["CRWD", "DDOG", BENCHMARK], ["DDOG"]]
    assert slept == [1.0]


def test_tickers_are_upper_cased_before_asking() -> None:
    source = Source(_frame({"CRWD": [1.0], BENCHMARK: [1.0]}))
    out = fetch(["crwd", "CRWD", "qqq"], source, now=AFTER_CLOSE, sleep=_no_sleep)
    assert source.calls == [(["CRWD", BENCHMARK], "1y")]
    assert out.notes == ()


def test_fetch_leaves_out_todays_unsettled_bar() -> None:
    days = pd.bdate_range(end="2026-10-12", periods=3)
    frame = _frame({"CRWD": [1.0, 2.0, 3.0], BENCHMARK: [1.0, 1.0, 1.0]}, days)
    frame[("Close", "NEW")] = [math.nan, math.nan, 5.0]
    open_session = datetime(2026, 10, 12, 19, 0, tzinfo=UTC)  # 15:00 New York
    source = Source(frame, _frame({"NEW": [5.0]}, days[-1:]), _frame({"NEW": [5.0]}, days[-1:]))
    out = fetch(["CRWD", "NEW"], source, now=open_session, sleep=_no_sleep)
    assert out.bars["CRWD"][-1].day == date(2026, 10, 9)
    assert out.bars[BENCHMARK][-1].day == date(2026, 10, 9)
    assert out.notes == ("no prices for NEW",)


def test_a_stock_whose_prices_stop_early_is_noted() -> None:
    frame = _frame({"CRWD": [1.0, 2.0, math.nan], "DDOG": [1.0, 2.0, 3.0], BENCHMARK: [1.0] * 3})
    out = fetch(["CRWD", "DDOG"], Source(frame), now=AFTER_CLOSE, sleep=_no_sleep)
    assert out.notes == (f"prices end before QQQ's {DAYS[2].date()}: CRWD ({DAYS[1].date()})",)


def test_the_benchmark_is_always_asked_for_once() -> None:
    source = Source(_frame({"CRWD": [1.0], BENCHMARK: [1.0]}))
    fetch(["CRWD", BENCHMARK, "CRWD"], source, now=AFTER_CLOSE, sleep=_no_sleep)
    assert source.calls == [(["CRWD", BENCHMARK], "1y")]


def test_a_failed_download_is_retried() -> None:
    slept: list[float] = []
    source = Source(OSError("reset"), pd.DataFrame(), _frame({"CRWD": [1.0], BENCHMARK: [1.0]}))
    out = fetch(["CRWD"], source, now=AFTER_CLOSE, sleep=slept.append)
    assert set(out.bars) == {"CRWD", BENCHMARK}
    assert out.notes == ()
    assert slept == [1.0, 2.0]


def test_a_download_that_never_works_fails_open() -> None:
    source = Source(*[RuntimeError("Yahoo says no, key=SECRET")] * prices.ATTEMPTS)
    out = fetch(["CRWD"], source, now=AFTER_CLOSE, sleep=_no_sleep)
    assert out.bars == {}
    assert out.notes == ("price download failed (RuntimeError)",)


def test_an_unreadable_frame_fails_open() -> None:
    class Odd:
        empty = False
        columns = property(lambda self: 1 / 0)

    source = Source(*[Odd()] * prices.ATTEMPTS)
    out = fetch(["CRWD"], source, now=AFTER_CLOSE, sleep=_no_sleep)
    assert out.bars == {}
    assert out.notes == ("price download failed (ZeroDivisionError)",)


OCT12, DEC14 = date(2026, 10, 12), date(2026, 12, 14)


@pytest.mark.parametrize(
    ("now", "last_day", "kept"),
    [
        (datetime(2026, 10, 12, 19, 0, tzinfo=UTC), OCT12, 1),  # 15:00 New York, session open
        (datetime(2026, 10, 12, 20, 29, tzinfo=UTC), OCT12, 1),  # 16:29, not settled
        (datetime(2026, 10, 12, 20, 30, tzinfo=UTC), OCT12, 2),  # 16:30, settled
        (datetime(2026, 10, 13, 3, 0, tzinfo=UTC), OCT12, 2),  # 23:00 the same day
        (datetime(2026, 10, 13, 14, 0, tzinfo=UTC), OCT12, 2),  # the next morning
        (datetime(2026, 12, 14, 21, 29, tzinfo=UTC), DEC14, 1),  # 16:29 in winter (UTC-5)
        (datetime(2026, 12, 14, 21, 30, tzinfo=UTC), DEC14, 2),  # 16:30 in winter
    ],
)
def test_todays_bar_waits_for_the_close(now: datetime, last_day: date, kept: int) -> None:
    bars = (Bar(date(2026, 10, 9), 1.0, None), Bar(last_day, 2.0, None))
    assert len(drop_unfinished(bars, now)) == kept


def test_trailing_return_needs_enough_closes() -> None:
    assert trailing_return([100.0, 110.0], 1) == pytest.approx(0.1)
    assert trailing_return([100.0, 110.0, 99.0], 2) == pytest.approx(-0.01)
    assert trailing_return([100.0, 110.0], 2) is None
    assert trailing_return([100.0], 0) is None


def test_volatility_matches_a_hand_calculation() -> None:
    closes = [100.0, 110.0, 99.0]
    logs = [math.log(1.1), math.log(0.9)]
    mean = sum(logs) / 2
    hand = math.sqrt(sum((x - mean) ** 2 for x in logs)) * math.sqrt(252)
    assert volatility(closes, 2) == pytest.approx(hand)
    assert volatility([100.0] * 21) == 0.0
    assert volatility([100.0] * 20) is None


def test_facts_compare_against_qqq_on_the_same_days() -> None:
    stock = [100.0 * 1.01**i for i in range(25)]
    bench = [50.0] * 25
    facts = price_facts("CRWD", Prices(_bars({"CRWD": stock, BENCHMARK: bench}), ()))
    assert facts is not None
    assert facts.day == DAYS[24].date()
    assert facts.close == pytest.approx(stock[-1])
    assert facts.returns[1] == pytest.approx(0.01)
    assert facts.returns[5] == pytest.approx(1.01**5 - 1)
    assert facts.returns[20] == pytest.approx(1.01**20 - 1)
    assert facts.benchmark_returns == {1: 0.0, 5: 0.0, 20: 0.0}
    assert facts.excess[20] == pytest.approx(1.01**20 - 1)
    assert facts.volatility == pytest.approx(0.0, abs=1e-12)
    assert facts.to_dict()["day"] == DAYS[24].date().isoformat()


def test_a_day_only_one_side_traded_is_skipped() -> None:
    days = [d.date() for d in DAYS[:4]]
    stock = tuple(Bar(d, c, None) for d, c in zip(days, [10.0, 11.0, 12.0, 13.0], strict=True))
    bench = tuple(Bar(d, c, None) for d, c in [(days[0], 1.0), (days[1], 1.0), (days[3], 2.0)])
    assert [s.day for s, _ in aligned(stock, bench)] == [days[0], days[1], days[3]]
    facts = price_facts("CRWD", Prices({"CRWD": stock, BENCHMARK: bench}, ()))
    assert facts is not None
    # 13 against 11 (the last shared day before), not against 12
    assert facts.returns[1] == pytest.approx(13 / 11 - 1)
    assert facts.excess[1] == pytest.approx(13 / 11 - 1 - 1.0)


def test_short_history_gives_none_not_a_guess() -> None:
    facts = price_facts("CRWD", Prices(_bars({"CRWD": [1.0, 2.0], BENCHMARK: [1.0, 1.0]}), ()))
    assert facts is not None
    assert facts.returns == {1: 1.0, 5: None, 20: None}
    assert facts.excess == {1: 1.0, 5: None, 20: None}
    assert facts.volatility is None


def test_no_facts_without_both_series() -> None:
    assert price_facts("CRWD", Prices(_bars({"CRWD": [1.0, 2.0]}), ())) is None
    assert price_facts("CRWD", Prices(_bars({BENCHMARK: [1.0, 2.0]}), ())) is None
    late: dict[str, tuple[Bar, ...]] = {
        "CRWD": (Bar(date(2026, 9, 1), 1.0, None),),
        BENCHMARK: (Bar(date(2026, 9, 2), 1.0, None),),
    }
    assert price_facts("CRWD", Prices(late, ())) is None


def test_the_cli_prints_facts_and_notes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    frame = _frame({"CRWD": [100.0, 101.0], BENCHMARK: [10.0, 10.0]})
    monkeypatch.setattr(prices, "YahooPrices", lambda: Source(frame))
    monkeypatch.setattr(prices, "datetime", _Clock)
    assert prices.main(["crwd", "gone"]) == 0
    out = capsys.readouterr().out
    assert "note: no prices for GONE" in out
    assert "CRWD   2026-09-02     101.00  vs QQQ: 1d +1.0%  5d n/a  20d n/a  vol n/a" in out
    assert "GONE   no prices" in out


class _Clock(datetime):
    @classmethod
    def now(cls, tz: Any = None) -> _Clock:
        return cls(2026, 9, 3, 23, 0, tzinfo=UTC)


def _bars(series: dict[str, Sequence[float]]) -> dict[str, tuple[Bar, ...]]:
    return {
        ticker: tuple(Bar(d.date(), c, None) for d, c in zip(DAYS, closes, strict=False))
        for ticker, closes in series.items()
    }
