from datetime import UTC, datetime

import pytest

from tenq import news
from tenq.news import (
    Entry,
    clean,
    collect,
    item_id,
    mentions,
    normalise_name,
    parse_feed,
)

SINCE = datetime(2026, 11, 3, 12, 0, tzinfo=UTC)
GENERAL = "https://general.example/rss"
YAHOO = "https://yahoo.example/{ticker}"


def _rss(*items: tuple[str, str, str | None, str]) -> bytes:
    """(title, link, RFC 822 date or None, summary) to an RSS 2.0 document."""
    body = []
    for title, link, date, summary in items:
        when = f"<pubDate>{date}</pubDate>" if date else ""
        body.append(
            f"<item><title>{title}</title><link>{link}</link>{when}"
            f"<description>{summary}</description></item>"
        )
    return (
        '<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>'
        + "".join(body)
        + "</channel></rss>"
    ).encode()


LATE = "Tue, 03 Nov 2026 20:00:00 GMT"
LATER = "Wed, 04 Nov 2026 09:00:00 GMT"
EARLY = "Tue, 03 Nov 2026 08:00:00 GMT"


class Feeds:
    def __init__(self, pages: dict[str, bytes | Exception]) -> None:
        self.pages = pages
        self.calls: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.calls.append(url)
        page = self.pages.get(url, _rss())
        if isinstance(page, Exception):
            raise page
        return page


def _collect(pages: dict[str, bytes | Exception], tickers: dict[str, str | None]) -> news.News:
    return collect(tickers, SINCE, fetch=Feeds(pages), general=[GENERAL], per_ticker=YAHOO)


@pytest.mark.parametrize(
    ("text", "ticker", "name", "hit"),
    [
        ("CRWD shares rise", "CRWD", None, True),
        ("Shares of $CRWD rise", "CRWD", None, True),
        ("crwd shares rise", "CRWD", None, False),  # symbols match in capitals only
        ("Net income rose at NETflix", "NET", None, False),
        ("Cloudflare (NET) jumps", "NET", None, True),
        ("NET-ZERO pledge", "NET", None, False),
        ("U.S. stocks slide", "S", None, False),  # short symbol needs a marker
        ("SentinelOne (S) slides", "S", None, True),
        ("NYSE: S slides", "S", None, True),
        ("Dynatrace DT beats", "DT", None, False),
        ("$DT beats", "DT", None, True),
        ("CrowdStrike Holdings, Inc. reports", "CRWD", "CrowdStrike Holdings, Inc.", True),
        ("crowdstrike's new kill switch", "CRWD", "CrowdStrike", True),
        ("CrowdStrikers hackathon", "CRWD", "CrowdStrike", False),
        ("Palo Alto weather", "PANW", "Palo Alto Networks", False),
        ("Palo Alto Networks buys", "PANW", "Palo Alto Networks", True),
        ("monday.com raises guidance", "MNDY", "monday.com", True),
    ],
)
def test_a_story_names_a_stock_only_by_its_symbol_or_company(
    text: str, ticker: str, name: str | None, hit: bool
) -> None:
    assert mentions(text, ticker, name) is hit


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("CrowdStrike Holdings, Inc.", "crowdstrike"),
        ("Datadog, Inc.", "datadog"),
        ("Box Inc.", "box inc"),  # stripping would leave a name shorter than MIN_NAME
        ("Palo Alto Networks", "palo alto networks"),
    ],
)
def test_legal_suffixes_are_dropped_from_company_names(name: str, key: str) -> None:
    assert normalise_name(name) == key


def test_headline_text_is_plain_and_capped() -> None:
    assert clean("<b>Up</b> &amp;   <i>away</i>\n", 100) == "Up & away"
    long = clean("word " * 200, 50)
    assert len(long) <= 50 and long.endswith("...")


def test_the_same_story_keeps_one_id_whatever_its_tracking_query() -> None:
    a = Entry("t", "https://x.example/story?utm=1", "f", None, "")
    b = Entry("t", "HTTPS://X.example/story?utm=2", "g", None, "")
    assert item_id("CRWD", a) == item_id("CRWD", b)
    assert item_id("CRWD", a).startswith("news:CRWD:")
    assert item_id("CRWD", a) != item_id("NET", a)


def test_general_and_per_ticker_feeds_are_matched_dated_and_deduped() -> None:
    pages: dict[str, bytes | Exception] = {
        GENERAL: _rss(
            ("CrowdStrike buys a startup", "https://a.example/1?x=1", LATE, ""),
            ("Cloudflare (NET) slides", "https://a.example/2", LATER, "peers fall"),
            ("Old CRWD news", "https://a.example/3", EARLY, ""),  # before `since`
            ("Undated CRWD news", "https://a.example/4", None, ""),  # no date
            ("Rates rise", "https://a.example/5", LATE, "markets"),  # names nobody
        ),
        YAHOO.format(ticker="CRWD"): _rss(
            ("CrowdStrike buys a startup", "https://a.example/1?x=2", LATE, ""),  # repeat
            ("A peer slides", "https://b.example/9", LATE, "Zscaler falls"),  # never names it
            ("Analysts on CRWD", "https://b.example/8", LATER, ""),
        ),
    }
    got = _collect(pages, {"CRWD": "CrowdStrike", "NET": "Cloudflare"})
    assert [i.title for i in got.items["CRWD"]] == [
        "Analysts on CRWD",
        "CrowdStrike buys a startup",
    ]  # newest first
    assert [i.title for i in got.items["NET"]] == ["Cloudflare (NET) slides"]
    assert got.items["NET"][0].summary == "peers fall"
    assert got.notes == ()


def test_a_repeated_headline_under_another_link_is_kept_once() -> None:
    pages: dict[str, bytes | Exception] = {
        GENERAL: _rss(("CRWD jumps", "https://a.example/1", LATE, "")),
        YAHOO.format(ticker="CRWD"): _rss(("CRWD JUMPS", "https://b.example/1", LATE, "")),
    }
    assert len(_collect(pages, {"CRWD": None}).items["CRWD"]) == 1


def test_items_are_capped_per_ticker() -> None:
    many = [(f"CRWD story {n}", f"https://a.example/{n}", LATE, "") for n in range(30)]
    got = collect(
        {"CRWD": None},
        SINCE,
        fetch=Feeds({GENERAL: _rss(*many)}),
        general=[GENERAL],
        per_ticker=None,
        cap=4,
    )
    assert len(got.items["CRWD"]) == 4


def test_a_dead_feed_is_a_note_and_the_rest_still_arrive() -> None:
    pages: dict[str, bytes | Exception] = {
        GENERAL: OSError("down"),
        YAHOO.format(ticker="CRWD"): _rss(("CRWD jumps", "https://a.example/1", LATE, "")),
        YAHOO.format(ticker="NET"): b"<<< not a feed",
    }
    feeds = Feeds(pages)
    got = collect(
        {"CRWD": None, "NET": None}, SINCE, fetch=feeds, general=[GENERAL], per_ticker=YAHOO
    )
    assert [i.title for i in got.items["CRWD"]] == ["CRWD jumps"]
    assert got.items["NET"] == ()
    assert got.notes == ("2 news feed(s) unavailable",)
    assert feeds.calls.count(GENERAL) == news.ATTEMPTS  # retried before giving up


def test_feed_text_is_cleaned_and_entries_without_titles_skipped() -> None:
    raw = _rss(
        ("&lt;b&gt;CRWD&lt;/b&gt; up", "https://a.example/1", LATE, "&lt;p&gt;Body&lt;/p&gt;"),
        ("", "https://a.example/2", LATE, "no title"),
    )
    entries = parse_feed(raw, GENERAL)
    assert [(e.title, e.summary) for e in entries] == [("CRWD up", "Body")]
    assert entries[0].published == datetime(2026, 11, 3, 20, 0, tzinfo=UTC)


def test_items_serialise_for_the_run_record() -> None:
    pages: dict[str, bytes | Exception] = {
        GENERAL: _rss(("CRWD jumps", "https://a.example/1", LATE, "s")),
    }
    item = _collect(pages, {"CRWD": None}).items["CRWD"][0]
    d = item.to_dict()
    assert d["published"] == "2026-11-03T20:00:00+00:00"
    assert d["id"] == item.id and d["ticker"] == "CRWD"


def test_the_cli_prints_counts_and_ids(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake(tickers: dict[str, str | None], since: datetime) -> news.News:
        assert tickers == {"CRWD": "CrowdStrike"}
        return news.News({"CRWD": ()}, ("1 news feed(s) unavailable",))

    monkeypatch.setattr(news, "collect", fake)
    assert news.main(["crwd"]) == 0
    out = capsys.readouterr().out
    assert "note: 1 news feed(s) unavailable" in out and "CRWD: 0" in out
