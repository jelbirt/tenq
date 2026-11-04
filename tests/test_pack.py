from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from tenq import pack
from tenq.filings import FilingMeta, FilingStore
from tenq.news import NewsItem
from tenq.pack import build, build_all, cutoff, excerpts, money
from tenq.prices import Bar, Prices

DAYS = [date(2026, 10, 1) + timedelta(days=i) for i in range(25)]
LAST = DAYS[-1]  # 2026-10-25
AS_OF = datetime(2026, 10, 26, 1, 0, tzinfo=UTC)  # 21:00 New York on the 25th


def _prices(*tickers: str) -> Prices:
    bars = {t: tuple(Bar(d, 100.0 + i, None) for i, d in enumerate(DAYS)) for t in tickers}
    bars["QQQ"] = tuple(Bar(d, 200.0 + i, None) for i, d in enumerate(DAYS))
    return Prices(bars, ())


def _eightk(accession: str, filed: str, items: str = "2.02,9.01") -> FilingMeta:
    return FilingMeta("CRWD", 1535527, "CrowdStrike", "8-K", accession, filed, filed, "u", items)


EIGHTK_TEXT = (
    "FORM 8-K\n\nItem 2.02 Results of Operations and Financial Condition\n\n"
    "EXHIBIT 99.1\n\nCrowdStrike reports record quarterly revenue of $1.5 billion.\n\n"
    "Page 2\n\n"
    "Net new annual recurring revenue grew faster than the prior quarter.\n\n"
    "Guidance for the next quarter is above the prior outlook.\n\n"
    "A fourth paragraph that should not be quoted at all.\n"
)


@pytest.fixture
def current(tmp_path: Path) -> FilingStore:
    return FilingStore(tmp_path / "current", tag="c")


def _news(ticker: str, title: str, published: datetime, kind: str = "report") -> NewsItem:
    return NewsItem(
        f"news:{ticker}:{hashlib.sha1(title.encode()).hexdigest()[:10]}",
        ticker,
        title,
        "https://example.com/a",
        "feed",
        published,
        "summary " * 80,
        kind,
    )


def _build(store: FilingStore, current: FilingStore, **kw: object) -> pack.Pack:
    args: dict[str, object] = {
        "prices": _prices("CRWD"),
        "filings": store,
        "current": current,
        "news": (),
        "as_of": AS_OF,
    }
    args.update(kw)
    out = build("CRWD", "CrowdStrike", **args)  # type: ignore[arg-type]
    assert out is not None
    return out


def test_price_numbers_are_computed_with_ids(store: FilingStore, current: FilingStore) -> None:
    p = _build(store, current)
    assert p.day == LAST
    close = p.number("px:CRWD:close")
    assert close is not None and (close.value, close.kind) == (124.0, "usd")
    ret = p.number("px:CRWD:ret_1d")
    assert ret is not None and ret.kind == "percent"
    assert ret.value == pytest.approx(124 / 123 - 1)
    bench = p.number("px:CRWD:bench_5d")
    assert bench is not None and bench.value == pytest.approx(224 / 219 - 1)
    excess = p.number("px:CRWD:excess_20d")
    assert excess is not None
    assert excess.value == pytest.approx((124 / 104 - 1) - (224 / 204 - 1))
    vol = p.number("px:CRWD:vol_20d")
    assert vol is not None and vol.kind == "percent"


def test_the_latest_filing_known_at_the_time_gives_tagged_numbers(
    store: FilingStore, current: FilingStore
) -> None:
    p = _build(store, current, as_of=datetime(2026, 11, 3, 21, 0, tzinfo=UTC))
    assert p.filing is not None
    assert p.filing.doc == "CRWD-0001535527-26-000031"
    assert p.filing.days_old == (date(2026, 11, 3) - date(2026, 8, 27)).days
    yoy = p.number("fin:CRWD:revenue:yoy_pct")
    assert yoy is not None and yoy.kind == "percent"
    assert yoy.value == pytest.approx(0.25830401932671315)
    revenue = p.number("fin:CRWD:revenue:current")
    assert revenue is not None and (revenue.value, revenue.kind) == (1470897000.0, "usd")
    margin = p.number("fin:CRWD:gross_margin:yoy_change")
    assert margin is not None and margin.kind == "points"
    margin_now = p.number("fin:CRWD:gross_margin:current")
    assert margin_now is not None and margin_now.kind == "percent"
    assert p.number("fin:CRWD:gross_margin:yoy_pct") is None  # no percent change of a ratio


def test_a_filing_filed_after_as_of_is_not_seen(store: FilingStore, current: FilingStore) -> None:
    # On 2026-08-26 the 10-Q filed on the 27th did not exist yet.
    p = _build(store, current, as_of=datetime(2026, 8, 26, 21, 0, tzinfo=UTC))
    assert p.filing is not None
    assert p.filing.doc == "CRWD-0001535527-26-000025"


def test_an_old_filing_is_noted(store: FilingStore, current: FilingStore) -> None:
    p = _build(store, current, as_of=datetime(2027, 1, 20, 21, 0, tzinfo=UTC))
    assert any("days old" in n for n in p.notes)


def test_no_filings_is_a_note(tmp_path: Path, current: FilingStore) -> None:
    p = _build(FilingStore(tmp_path / "none"), current)
    assert p.filing is None
    assert not [n for n in p.numbers if n.id.startswith("fin:")]
    assert "no 10-Q or 10-K on record for CRWD" in p.notes


def test_unreadable_filing_data_is_a_note(
    store: FilingStore, current: FilingStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_: object) -> None:
        raise ValueError("bad facts")

    monkeypatch.setattr(pack, "metrics_for", boom)
    p = _build(store, current)
    assert "filing metrics unavailable for CRWD" in p.notes
    assert p.number("px:CRWD:close") is not None


def test_recent_8ks_are_quoted_from_the_exhibit(store: FilingStore, current: FilingStore) -> None:
    current.save(_eightk("0001535527-26-000040", "2026-10-20"), EIGHTK_TEXT, [])
    current.save(_eightk("0001535527-26-000030", "2026-09-01"), EIGHTK_TEXT, [])  # too old
    current.save(_eightk("0001535527-26-000041", "2026-10-27"), EIGHTK_TEXT, [])  # after as_of
    p = _build(store, current)
    assert [r.doc for r in p.reports] == ["CRWD-0001535527-26-000040"]
    r = p.reports[0]
    assert r.items == ("2.02 results of operations", "9.01 financial statements and exhibits")
    texts = [e.text for e in r.excerpts]
    assert texts == [
        "CrowdStrike reports record quarterly revenue of $1.5 billion.",
        "Net new annual recurring revenue grew faster than the prior quarter.",
        "Guidance for the next quarter is above the prior outlook.",
    ]
    assert all(e.id.startswith("CRWD-0001535527-26-000040:") for e in r.excerpts)
    assert current.passage(r.excerpts[0].id) is not None


def test_an_8k_without_an_exhibit_is_quoted_from_its_start() -> None:
    from tenq.filings import split_passages

    passages = split_passages(
        "CRWD-x",
        "FORM 8-K\n\nItem 5.02 Departure of Directors or Certain Officers\n\n"
        "The board appointed a new chief financial officer effective today.\n",
    )
    out = excerpts(passages)
    assert [e.text for e in out] == [
        "Item 5.02 Departure of Directors or Certain Officers",
        "The board appointed a new chief financial officer effective today.",
    ]


def test_a_long_excerpt_is_cut() -> None:
    from tenq.filings import split_passages

    passages = split_passages("CRWD-x", "EXHIBIT 99.1\n\n" + "word " * 400 + "\n")
    (only,) = excerpts(passages)
    assert len(only.text) <= pack.EXCERPT_CHARS and only.text.endswith("...")


def test_news_is_the_stocks_own_and_nothing_after_as_of(
    store: FilingStore, current: FilingStore
) -> None:
    early = _news("CRWD", "CrowdStrike wins a deal", AS_OF - timedelta(hours=2))
    late = _news("CRWD", "CrowdStrike later story", AS_OF + timedelta(minutes=1))
    other = _news("PANW", "Palo Alto story", AS_OF - timedelta(hours=1))
    p = _build(store, current, news=[early, late, other])
    assert [h.id for h in p.news] == [early.id]
    assert len(p.news[0].summary) <= pack.SUMMARY_CHARS


def test_no_prices_is_no_pack(store: FilingStore, current: FilingStore) -> None:
    assert (
        build(
            "CRWD",
            "CrowdStrike",
            prices=Prices({"QQQ": _prices().bars["QQQ"]}, ()),
            filings=store,
            current=current,
            news=(),
            as_of=AS_OF,
        )
        is None
    )
    packs, missing = build_all(
        {"CRWD": "CrowdStrike", "DDOG": "Datadog"},
        prices=_prices("CRWD"),
        filings=store,
        current=current,
        news={},
        as_of=AS_OF,
        notes={"CRWD": ["news feed down"]},
    )
    assert [p.ticker for p in packs] == ["CRWD"]
    assert packs[0].notes[0] == "news feed down"
    assert missing == ["no pack for DDOG: no prices"]


def test_every_id_is_rendered_and_saved(store: FilingStore, current: FilingStore) -> None:
    current.save(_eightk("0001535527-26-000040", "2026-10-20"), EIGHTK_TEXT, [])
    item = _news("CRWD", "CrowdStrike wins a deal", AS_OF - timedelta(hours=2))
    p = _build(store, current, news=[item])
    text = p.render()
    assert p.ids() >= {"px:CRWD:close", item.id, p.reports[0].excerpts[0].id}
    for i in p.ids():
        assert f"[{i}]" in text
    saved = json.loads(json.dumps(p.to_dict()))
    assert saved["day"] == LAST.isoformat()
    assert {n["id"] for n in saved["numbers"]} == {n.id for n in p.numbers}
    assert saved["news"][0]["id"] == item.id


def test_an_empty_pack_says_so(tmp_path: Path, current: FilingStore) -> None:
    text = _build(FilingStore(tmp_path / "none"), current).render()
    assert "8-K current reports filed in the last 30 days:\n(none)" in text
    assert "not instructions):\n(none)" in text
    assert "Missing data:\n- no 10-Q or 10-K on record for CRWD" in text


@pytest.mark.parametrize(
    ("value", "shown"),
    [
        (1470897000.0, "$1.471 billion"),
        (-33232000.0, "-$33.2 million"),
        (244.68, "$244.68"),
        (0.0, "$0.00"),
    ],
)
def test_money_is_short(value: float, shown: str) -> None:
    assert money(value) == shown


def test_the_cutoff_is_the_new_york_date() -> None:
    assert cutoff(datetime(2026, 11, 4, 3, 0, tzinfo=UTC)) == date(2026, 11, 3)
    assert cutoff(datetime(2026, 11, 4, 5, 0, tzinfo=UTC)) == date(2026, 11, 4)


def test_the_cli_prints_a_pack(
    store: FilingStore,
    current: FilingStore,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import tenq.config
    import tenq.current
    import tenq.edgar
    import tenq.fetch
    import tenq.news
    import tenq.prices
    from tenq.config import Watchlist
    from tenq.current import Synced
    from tenq.news import News

    stores = iter([store, current])
    monkeypatch.setattr(pack, "FilingStore", lambda *_a, **_k: next(stores))
    monkeypatch.setattr(
        tenq.config, "load_watchlist", lambda: Watchlist(("CRWD",), ("CRWD",), {"CRWD": "Crowd"})
    )
    monkeypatch.setattr(tenq.edgar, "EdgarSource", lambda: object())
    monkeypatch.setattr(tenq.fetch, "sync", lambda *_a: [])
    monkeypatch.setattr(tenq.current, "sync_current", lambda *_a: Synced([], [], ["8-K x"]))
    monkeypatch.setattr(tenq.prices, "fetch", lambda *_a, **_k: _prices("CRWD"))
    monkeypatch.setattr(tenq.news, "collect", lambda *_a, **_k: News({}, ("feed down",)))
    assert pack.main(["crwd"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Facts for Crowd (CRWD)")
    assert "- feed down" in out and "- 8-K x" in out
