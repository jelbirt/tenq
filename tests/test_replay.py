from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from tenq import replay
from tenq.prices import BENCHMARK, Bar, Prices
from tenq.scorecard import day_totals, is_session, load_config, standard_error, t95

CFG = load_config()


def _days(n: int, start: date = date(2025, 1, 2)) -> list[date]:
    out: list[date] = []
    d = start
    while len(out) < n:
        if is_session(d):
            out.append(d)
        d += timedelta(days=1)
    return out


def _bars(days: list[date], growth: float) -> tuple[Bar, ...]:
    """A price that rises by `growth` a session, opening at the last close."""
    out, price = [], 100.0
    for d in days:
        out.append(Bar(d, price * (1 + growth), None, open=price))
        price *= 1 + growth
    return tuple(out)


def _prices(n: int = 60, stock: float = 0.01, start: int = 0) -> Prices:
    days = _days(n)
    return Prices({BENCHMARK: _bars(days, 0.0), "ACME": _bars(days[start:], stock)}, ())


def test_every_call_is_scored_like_a_live_call_from_the_next_open() -> None:
    rows = replay.rows_from(_prices(), ["ACME"], CFG)
    days = _days(60)
    # a call made on day i enters at day i+1's open and exits 20 sessions on
    assert [r["day"] for r in rows] == [d.isoformat() for d in days[: 60 - CFG.sessions]]
    first = rows[0]
    assert first["exit_day"] == days[CFG.sessions].isoformat()
    assert first["excess"] == pytest.approx(1.01**20 - 1)
    assert sum(first["moves"]) == pytest.approx(first["excess"])


def test_momentum_uses_only_the_20_sessions_up_to_that_close() -> None:
    rows = replay.rows_from(_prices(), ["ACME"], CFG)
    assert all(r["momentum"] is None for r in rows[: replay.LOOKBACK])
    assert rows[replay.LOOKBACK]["momentum"] == pytest.approx(1.01**20 - 1)


def _jump(n: int, k: int) -> Prices:
    """Flat prices, except the stock gaps up 10% at day k's open and adds 10% more by its close."""
    days = _days(n)
    stock = []
    for i, d in enumerate(days):
        if i < k:
            stock.append(Bar(d, 100.0, None, open=100.0))
        elif i == k:
            stock.append(Bar(d, 121.0, None, open=110.0))
        else:
            stock.append(Bar(d, 121.0, None, open=121.0))
    return Prices({BENCHMARK: _bars(days, 0.0), "ACME": tuple(stock)}, ())


def test_a_call_cannot_see_the_next_day() -> None:
    k = 30
    rows = {r["day"]: r for r in replay.rows_from(_jump(60, k), ["ACME"], CFG)}
    days = _days(60)
    before = rows[days[k - 1].isoformat()]
    # made the evening before the jump: momentum has not seen it ...
    assert before["momentum"] == 0.0
    # ... and the call enters at the jump day's open, after the gap, before the day's rise
    assert before["excess"] == pytest.approx(121 / 110 - 1)
    # the evening of the jump, momentum shows it
    assert rows[days[k].isoformat()]["momentum"] == pytest.approx(0.21)


def test_momentum_counts_the_days_both_traded_as_the_live_sheet_does() -> None:
    days = _days(60)
    stock = tuple(b for i, b in enumerate(_bars(days, 0.01)) if i != 25)  # one missing bar
    prices = Prices({BENCHMARK: _bars(days, 0.0), "ACME": stock}, ())
    rows = replay.rows_from(prices, ["ACME"], CFG)
    assert days[25].isoformat() not in {r["day"] for r in rows}  # no call without a bar
    after = next(r for r in rows if r["day"] == days[30].isoformat())
    # 20 shared days back from day 30 reach day 9, past the gap: 21 sessions of 1% growth
    assert after["momentum"] == pytest.approx(1.01**21 - 1)


def test_the_luck_range_comes_from_daily_moves_not_rows() -> None:
    days = _days(80)
    noisy = []
    price = 100.0
    for i, d in enumerate(days):
        step = 0.03 if i % 3 == 0 else -0.012
        noisy.append(Bar(d, price * (1 + step), None, open=price))
        price *= 1 + step
    prices = Prices({BENCHMARK: _bars(days, 0.0), "ACME": tuple(noisy)}, ())
    rows = replay.rows_from(prices, ["ACME"], CFG)
    got = replay.summary(rows, "always_buy", CFG)
    pairs = [(1.0, r) for r in rows]
    totals = list(day_totals(pairs).values())
    se = standard_error(totals, len(pairs), CFG.carry_days)
    assert se is not None
    assert got["luck"] == pytest.approx(t95(len(totals) - 1) * se)


def test_a_stock_listed_later_starts_when_its_prices_do() -> None:
    rows = replay.rows_from(_prices(start=10), ["ACME"], CFG)
    assert rows[0]["day"] == _days(60)[10].isoformat()


def test_the_baselines_score_as_the_scorecard_does() -> None:
    rows = replay.rows_from(_prices(stock=0.01), ["ACME"], CFG)
    buy = replay.summary(rows, "always_buy", CFG)
    hold = replay.summary(rows, "always_hold", CFG)
    winners = replay.summary(rows, "last_months_winners", CFG)
    assert buy["mean"] == pytest.approx(1.01**20 - 1) and buy["hit"] == 1.0
    assert hold["mean"] == 0.0 and hold["buys"] == 0
    # no momentum for the first 20 days: no call then, a Buy on every day after
    assert winners["calls"] == len(rows) - replay.LOOKBACK == winners["buys"]


def test_a_falling_stock_makes_last_months_winners_hold() -> None:
    rows = replay.rows_from(_prices(stock=-0.01), ["ACME"], CFG)
    winners = replay.summary(rows, "last_months_winners", CFG)
    assert winners["buys"] == 0 and winners["mean"] == 0.0


def test_main_prints_the_table_and_writes_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(replay, "load_watchlist", lambda: _Watch(("ACME",)))
    out = tmp_path / "r.json"
    assert replay.main(["--json", str(out)], prices=lambda t, p: _prices()) == 0
    text = capsys.readouterr().out
    assert "Always Buy" in text and "Last month's winners" in text
    assert "flatters every rule that mostly buys" in text and "not the bar" in text
    assert json.loads(out.read_text())["rows"] == 60 - CFG.sessions


def test_main_without_qqq_prices_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(replay, "load_watchlist", lambda: _Watch(("ACME",)))
    assert replay.main([], prices=lambda t, p: Prices({}, ("no data",))) == 1


class _Watch:
    def __init__(self, tickers: tuple[str, ...]) -> None:
        self.tickers = tickers
