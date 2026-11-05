from datetime import UTC, datetime
from pathlib import Path

import pytest

from tenq import news
from tenq.news import (
    Entry,
    clean,
    collect,
    host,
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
        ("BUY NOW: 3 STOCKS TO WATCH", "NOW", "ServiceNow", False),
        ("S&P 500 NOW at record", "NOW", "ServiceNow", False),
        ("ServiceNow (NOW) beats", "NOW", "ServiceNow", True),
        ("GAAP NET LOSS narrowed", "NET", "Cloudflare", False),
        ("TEAM USA wins gold", "TEAM", "Atlassian", False),
        ("SHOP SMALL weekend", "SHOP", "Shopify", False),
        ("AI and IOT spending", "IOT", "Samsara", False),
        ("$SNOW rallies", "SNOW", "Snowflake", True),
        ("U.CRWD fund", "CRWD", None, False),  # a 3+ letter symbol after a dot
        ("$$CRWD", "CRWD", None, True),
        ("X$CRWD", "CRWD", None, False),
        ("CrowdStrike\u2019s guidance", "CRWD", "CrowdStrike", True),
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


def test_the_same_story_keeps_one_id_whatever_its_tracking_parameters() -> None:
    a = Entry("t", "https://x.example/story?utm_source=a&id=7", "f", None, "")
    b = Entry("t", "HTTPS://X.example/story?id=7&utm_medium=b&.tsrc=rss&ncid=1", "g", None, "")
    assert item_id("CRWD", a) == item_id("CRWD", b)
    assert item_id("CRWD", a).startswith("news:CRWD:")
    assert item_id("CRWD", a) != item_id("NET", a)


def test_a_query_that_names_the_story_keeps_stories_apart() -> None:
    one = Entry("t", "https://x.example/article.php?id=1", "f", None, "")
    two = Entry("t", "https://x.example/article.php?id=2", "f", None, "")
    assert item_id("CRWD", one) != item_id("CRWD", two)


def test_a_story_without_a_link_is_keyed_by_its_title() -> None:
    a = Entry("CRWD Jumps", "", "f", None, "")
    b = Entry("crwd jumps", "", "g", None, "")
    assert item_id("CRWD", a) == item_id("CRWD", b)


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


def test_a_story_counts_only_for_stocks_its_headline_names() -> None:
    pages: dict[str, bytes | Exception] = {
        GENERAL: _rss(
            ("Zscaler falls after earnings", "https://a.example/1", LATE, "peers CrowdStrike fell"),
        ),
        YAHOO.format(ticker="CRWD"): _rss(
            ("Security stocks slide", "https://b.example/2", LATE, "CrowdStrike fell 3%"),
        ),
    }
    got = _collect(pages, {"CRWD": "CrowdStrike", "ZS": "Zscaler"})
    assert [i.title for i in got.items["ZS"]] == ["Zscaler falls after earnings"]
    assert got.items["CRWD"] == ()  # named only in the summary, even in its own feed


def test_a_story_dated_exactly_at_since_is_not_new() -> None:
    at = "Tue, 03 Nov 2026 12:00:00 GMT"  # == SINCE
    pages: dict[str, bytes | Exception] = {GENERAL: _rss(("CRWD jumps", "https://a/1", at, ""))}
    assert _collect(pages, {"CRWD": None}).items["CRWD"] == ()


def test_an_updated_date_stands_in_for_a_missing_published_date() -> None:
    raw = (
        b'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><title>t</title>'
        b"<entry><title>CRWD up</title><link href='https://a.example/1'/>"
        b"<updated>2026-11-03T20:00:00Z</updated></entry></feed>"
    )
    (entry,) = parse_feed(raw, GENERAL)
    assert entry.published == datetime(2026, 11, 3, 20, 0, tzinfo=UTC)


def test_a_feed_body_is_never_read_as_a_file_path(tmp_path: Path) -> None:
    local = tmp_path / "local.xml"
    local.write_bytes(_rss(("LOCAL FILE CRWD", "https://a/1", LATE, "")))
    with pytest.raises(ValueError):
        parse_feed(str(local).encode(), GENERAL)


def test_a_hanging_per_ticker_feed_is_given_up_after_a_few_failures() -> None:
    tickers: dict[str, str | None] = {t: None for t in ("AAA", "BBB", "CCC", "DDD", "EEE")}
    feeds = Feeds({YAHOO.format(ticker=t): TimeoutError("hang") for t in tickers})
    got = collect(tickers, SINCE, fetch=feeds, general=[], per_ticker=YAHOO)
    assert len(set(feeds.calls)) == news.PER_TICKER_FAILURES
    assert got.notes == ("5 news feed(s) unavailable",)


def test_a_working_feed_resets_the_failure_count() -> None:
    tickers: dict[str, str | None] = {t: None for t in ("AAA", "BBB", "CCC", "DDD", "EEE")}
    pages: dict[str, bytes | Exception] = {
        YAHOO.format(ticker=t): TimeoutError("hang") for t in ("AAA", "BBB", "DDD", "EEE")
    }
    feeds = Feeds(pages)
    collect(tickers, SINCE, fetch=feeds, general=[], per_ticker=YAHOO)
    assert len(set(feeds.calls)) == 5


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
    assert d["id"] == item.id and d["ticker"] == "CRWD" and d["link"] == "https://a.example/1"
    assert "kind" not in d  # stories are not labelled; the models judge the source


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


def test_the_cap_keeps_the_newest_stories_whatever_their_site() -> None:
    columns = [
        (
            f"Is CrowdStrike a Buy? Take {n}",
            f"https://www.fool.com/{n}",
            f"Wed, 04 Nov 2026 0{n}:00:00 GMT",
            "",
        )
        for n in range(5)
    ]
    release = (
        "CrowdStrike Announces Third Quarter Results",
        "https://finance.yahoo.com/r",
        LATE,
        "",
    )
    got = collect(
        {"CRWD": "CrowdStrike"},
        SINCE,
        fetch=Feeds({GENERAL: _rss(release, *columns)}),
        general=[GENERAL],
        per_ticker=None,
        cap=3,
    )
    assert [i.title for i in got.items["CRWD"]] == [
        "Is CrowdStrike a Buy? Take 4",
        "Is CrowdStrike a Buy? Take 3",
        "Is CrowdStrike a Buy? Take 2",
    ]


def test_items_are_newest_first() -> None:
    pages: dict[str, bytes | Exception] = {
        GENERAL: _rss(
            ("CRWD falls", "https://a.example/1", LATE, ""),
            ("CRWD rises", "https://a.example/2", LATER, ""),
        )
    }
    assert [i.title for i in _collect(pages, {"CRWD": None}).items["CRWD"]] == [
        "CRWD rises",
        "CRWD falls",
    ]


def test_video_and_podcast_hosts_are_dropped() -> None:
    pages: dict[str, bytes | Exception] = {
        YAHOO.format(ticker="HUBS"): _rss(
            ("HubSpot founders on growth", "https://thehustle.videonest.co/1", LATE, ""),
            ("HubSpot beats estimates", "https://www.cnbc.com/2", LATE, ""),
        )
    }
    got = _collect(pages, {"HUBS": "HubSpot"})
    assert [i.title for i in got.items["HUBS"]] == ["HubSpot beats estimates"]


@pytest.mark.parametrize(
    ("link", "site"),
    [
        ("https://www.fool.com/x", "fool.com"),
        ("https://m.fool.com/x", "m.fool.com"),
        ("https://finance.yahoo.com/x", "finance.yahoo.com"),
        ("", ""),
        ("http://[oops/x", ""),
    ],
)
def test_a_story_s_site_is_its_link_s_host(link: str, site: str) -> None:
    assert host(link) == site


def test_a_look_alike_host_is_not_dropped() -> None:
    pages: dict[str, bytes | Exception] = {
        GENERAL: _rss(("CRWD on video", "https://notvideonest.co/1", LATE, "")),
    }
    assert [i.title for i in _collect(pages, {"CRWD": None}).items["CRWD"]] == ["CRWD on video"]


def test_a_link_that_is_not_a_url_never_sinks_the_run() -> None:
    pages: dict[str, bytes | Exception] = {
        GENERAL: _rss(
            ("Zscaler falls", "http://[oops/x", LATE, ""),
            ("CRWD rises", "http://[oops/y", LATE, ""),
        )
    }
    got = _collect(pages, {"CRWD": "CrowdStrike", "ZS": "Zscaler"})
    assert [i.title for i in got.items["CRWD"]] == ["CRWD rises"]
    assert [i.title for i in got.items["ZS"]] == ["Zscaler falls"]
