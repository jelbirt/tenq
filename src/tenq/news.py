"""News headlines for each watchlist stock, from RSS feeds, each with a stable id.

Two kinds of feed: Yahoo's per-ticker feed, and general business feeds (CNBC,
MarketWatch). Both carry stories about other companies: Yahoo's feed for a stock
lists its peers' news, and a summary names rivals in passing ("peers CrowdStrike and
Palo Alto also fell"). So a story counts for a stock only when its headline names
the stock by symbol or company name, with the rules ticker-sentinel learned the hard
way (`src/sentinel/news/matching.py`, MIT, same author): a symbol matches only in
capitals, so "net income" is never NET;
a one or two letter symbol, or one that is an English word (NOW, NET, TEAM), needs
a cashtag, brackets or an exchange prefix, so neither the "S" in "U.S." nor "BUY
NOW" is a stock; a company name matches without its legal suffix.

Each item is labelled by kind, so a reader can weigh a company's own announcement
above a columnist's take: KIND_COMPANY (the company's own press release), KIND_REPORT
(news reporting), KIND_RELEASE (another company's press release that names it) and
KIND_OPINION (buy-or-sell columns, ratings templates, stock-picking sites). The label
comes from the link's site, the headline's wording and a press release's dateline;
it is a heuristic, and a story it cannot place is a report. The per-ticker cap keeps
the highest kinds first, then the newest, so a busy day's clickbait cannot push out
the company's own release. Video and podcast hosts are dropped outright.

Headline text is data from the open web, never instructions. Nothing here raises: a
dead feed is skipped with a note.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import logging
import re
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

log = logging.getLogger(__name__)

YAHOO_FEED = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
GENERAL_FEEDS = (
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",  # CNBC top news
    "https://feeds.content.dowjones.io/public/rss/mw_topstories",  # MarketWatch
)
USER_AGENT = "tenq/0.1 (+https://github.com/jelbirt/tenq)"
TIMEOUT_S = 20
MAX_FEED_BYTES = 5_000_000
ATTEMPTS = 2
PER_TICKER = 10
MAX_TITLE = 300
MAX_SUMMARY = 600

SHORT_SYMBOL = 2
# symbols that are also English words get written in capitals in headlines ("BUY NOW",
# "GAAP NET LOSS", "TEAM USA"), so like short symbols they need a marker
WORD_SYMBOLS = frozenset({"NOW", "NET", "TEAM", "SHOP", "SNOW", "IOT"})
TRACKING = re.compile(r"(?:utm_.*|mod|\.tsrc|ncid|guccounter|guce_.*|cmpid|fbclid|gclid)$", re.I)
PER_TICKER_FAILURES = 3  # consecutive dead per-ticker feeds before the rest are skipped
MIN_NAME = 4
EXCHANGES = r"NYSE(?: American)?|NASDAQ|Nasdaq|AMEX"
LEGAL_SUFFIXES = frozenset(
    {"inc", "incorporated", "corp", "corporation", "holdings", "holding", "ltd", "limited"}
    | {"plc", "co", "company", "nv", "sa", "ag", "se"}
)

KIND_COMPANY = "company release"
KIND_REPORT = "report"
KIND_RELEASE = "release"
KIND_OPINION = "opinion"
KIND_RANK = {KIND_COMPANY: 0, KIND_REPORT: 1, KIND_RELEASE: 2, KIND_OPINION: 3}
# sites whose items are stock-picking columns, ratings or chatter rather than reporting
OPINION_HOSTS = frozenset(
    {"fool.com", "tikr.com", "trefis.com", "247wallst.com", "barchart.com", "stocktwits.com"}
    | {"marketbeat.com", "investorshub.advfn.com", "beincrypto.com", "seekingalpha.com"}
    | {"zacks.com", "simplywall.st", "insidermonkey.com"}
)
# sites with no reporting at all (HubSpot's podcasts, filed under HUBS by Yahoo)
DROP_HOSTS = frozenset({"videonest.co"})
# headline shapes of columns and ratings templates that Yahoo hosts on its own site
OPINION_TITLE = re.compile(
    r"\b(?:buy|sell|hold)\b[^.]{0,30}\?|better buy|should you|is it (?:time|a buy)"
    r"|time to (?:buy|sell|hold)|facts to (?:know|note)|bigger fall than|as market (?:rises|dips)"
    r"|stocks? to (?:buy|watch)|fair value|overvalued|undervalued|price i'?d pay"
    r"|what'?s in (?:store|the cards)|beat estimates again|read this before|worth the wait"
    r"|billionaire|\bvs\.?\s|top analyst calls|brokers suggest|could derail"
    # stock-site templates syndicated through Yahoo, whose links hide the site
    r"|\bzacks\b|here[\u2019']?s why|don[\u2019']?t overlook|compared to estimates|on the horizon"
    r"|look at key metrics|doubters",
    re.IGNORECASE,
)
# a market-research firm's promotion that lists companies in its headline
RESEARCH_TITLE = re.compile(
    r"\bmarket (?:report|size|share|forecast|outlook|analysis)\b|\bmarket 20[0-9]{2}\b",
    re.IGNORECASE,
)
# the transcript or summary of an earnings call reports what was said, whatever the site
CALL_TITLE = re.compile(r"earnings call (?:transcript|summary|highlights)", re.IGNORECASE)
# a press release opens with a dateline ("AUSTIN, Texas, November 04, 2026--") or a
# wire service's mark, says it was announced today, or describes itself after its
# listing ("(NASDAQ: PANW), the global ..."); reporters write "(NASDAQ:CRWD) fell"
RELEASE_TEXT = re.compile(
    r"^[^\n]{0,120}?\b20[0-9]{2}\s*(?:--|\(GLOBE NEWSWIRE\)|/PRNewswire/|\(BUSINESS WIRE\))"
    r"|\b(?:today announced|announced today)\b"
    r"|^[^\n]{0,80}?\((?:NASDAQ|Nasdaq|NYSE)(?: Global Select Market)?: [A-Z.]{1,6}\),"
    r"\s+(?:the|a|an)\s"
)
# with no release text, only a results announcement in the company's own words
# ("Datadog Announces Third Quarter 2026 Financial Results"); a reporter's "CrowdStrike
# launches AI kill switch" is a report
OWN_RESULTS = re.compile(r"\s+(?:Announces|Reports)\b.*\bResults\b")
TAG = re.compile(r"<[^>]*>")
SPACE = re.compile(r"\s+")

Fetch = Callable[[str], bytes]


@dataclass(frozen=True)
class NewsItem:
    id: str  # "news:CRWD:<10 hex>", stable for the same story
    ticker: str
    title: str
    link: str
    source: str  # the feed it came from
    published: datetime
    summary: str
    kind: str  # one of KIND_RANK

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "ticker": self.ticker,
            "title": self.title,
            "link": self.link,
            "source": self.source,
            "published": self.published.isoformat(),
            "summary": self.summary,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class Entry:
    title: str
    link: str
    source: str
    published: datetime | None
    summary: str


@dataclass(frozen=True)
class News:
    items: dict[str, tuple[NewsItem, ...]]
    notes: tuple[str, ...]


def http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        body: bytes = resp.read(MAX_FEED_BYTES + 1)
    if len(body) > MAX_FEED_BYTES:
        raise ValueError("feed too large")
    return body


def clean(text: str, limit: int) -> str:
    """Plain text: tags dropped, entities decoded, whitespace collapsed, capped."""
    out = SPACE.sub(" ", html.unescape(TAG.sub(" ", text))).strip()
    return out if len(out) <= limit else out[: limit - 3].rstrip() + "..."


def _published(entry: Mapping[str, Any]) -> datetime | None:
    stamp = entry.get("published_parsed") or entry.get("updated_parsed")
    if not stamp:
        return None
    try:
        year, month, day, hour, minute, second = (int(v) for v in stamp[:6])
        return datetime(year, month, day, hour, minute, second, tzinfo=UTC)
    except (TypeError, ValueError):
        return None


def parse_feed(raw: bytes, source: str) -> list[Entry]:
    import feedparser

    # feedparser treats a bytes argument as a path or URL first; a body from the open
    # web must never be read as a local file name
    parsed = feedparser.parse(io.BytesIO(raw))
    if parsed.bozo and not parsed.entries:
        raise ValueError("unreadable feed")
    out = []
    for e in parsed.entries:
        title = clean(str(e.get("title", "")), MAX_TITLE)
        if not title:
            continue
        out.append(
            Entry(
                title=title,
                link=str(e.get("link", "")).strip(),
                source=source,
                published=_published(e),
                summary=clean(str(e.get("summary", "")), MAX_SUMMARY),
            )
        )
    return out


def symbol_pattern(ticker: str) -> re.Pattern[str]:
    symbol = re.escape(ticker)
    if len(ticker) <= SHORT_SYMBOL or ticker in WORD_SYMBOLS:
        return re.compile(rf"(?:(?<!\w)\${symbol}\b|\({symbol}\)|\b(?:{EXCHANGES}):\s*{symbol}\b)")
    return re.compile(rf"(?<![\w$.-]){symbol}(?![\w-])|(?<!\w)\${symbol}\b")


def normalise_name(name: str) -> str:
    """ "CrowdStrike Holdings, Inc." to "crowdstrike"; never shorter than MIN_NAME."""
    tokens = [t for t in (w.strip(",.") for w in name.lower().split()) if t]
    while len(tokens) > 1 and tokens[-1].replace(".", "") in LEGAL_SUFFIXES:
        shorter = " ".join(tokens[:-1])
        if len(shorter) < MIN_NAME:
            break
        tokens = tokens[:-1]
    return " ".join(tokens)


def mentions(text: str, ticker: str, name: str | None) -> bool:
    if symbol_pattern(ticker).search(text):
        return True
    if not name:
        return False
    key = normalise_name(name)
    if len(key) < MIN_NAME:
        return False
    return re.search(rf"(?<!\w){re.escape(key)}(?!\w)", text.lower()) is not None


def host(link: str) -> str:
    """The link's site without "www.": "https://www.fool.com/x" to "fool.com"."""
    return (urlsplit(link).hostname or "").removeprefix("www.")


def _on(site: str, hosts: frozenset[str]) -> bool:
    return any(site == h or site.endswith("." + h) for h in hosts)


def kind_of(entry: Entry, ticker: str, name: str | None) -> str:
    """Which kind of item a story is for this stock; see the module docstring."""
    if CALL_TITLE.search(entry.title):
        return KIND_REPORT
    if _on(host(entry.link), OPINION_HOSTS) or OPINION_TITLE.search(entry.title):
        return KIND_OPINION
    if RESEARCH_TITLE.search(entry.title):
        return KIND_RELEASE
    own = (
        bool(name)
        and len(key := normalise_name(name or "")) >= MIN_NAME
        and entry.title.lower().startswith(key)
    )
    if RELEASE_TEXT.search(entry.summary):
        return KIND_COMPANY if own else KIND_RELEASE
    if own and OWN_RESULTS.match(entry.title[len(key) :]):
        return KIND_COMPANY
    return KIND_REPORT


def link_key(entry: Entry) -> str:
    """Same story, same key: a link without its tracking parameters, else the title."""
    if entry.link:
        parts = urlsplit(entry.link)
        query = urlencode(
            sorted((k, v) for k, v in parse_qsl(parts.query) if not TRACKING.match(k))
        )
        return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, query, ""))
    return entry.title.lower()


def item_id(ticker: str, entry: Entry) -> str:
    return f"news:{ticker}:{hashlib.sha1(link_key(entry).encode()).hexdigest()[:10]}"


def _read(url: str, fetch: Fetch) -> list[Entry] | None:
    for attempt in range(ATTEMPTS):
        try:
            return parse_feed(fetch(url), url)
        except Exception as exc:
            log.warning("feed attempt %d failed: %s (%s)", attempt + 1, url, type(exc).__name__)
    return None


def collect(
    tickers: Mapping[str, str | None],
    since: datetime,
    *,
    fetch: Fetch = http_get,
    general: Iterable[str] = GENERAL_FEEDS,
    per_ticker: str | None = YAHOO_FEED,
    cap: int = PER_TICKER,
) -> News:
    """Headlines published after `since` for each ticker (mapped to its company name).

    Undated entries are left out: with no date there is no telling whether a story is
    new today or has been in the feed for a week.
    """
    notes: list[str] = []
    found: dict[str, dict[str, NewsItem]] = {t: {} for t in tickers}
    undated = 0

    def add(ticker: str, entry: Entry) -> None:
        nonlocal undated
        if _on(host(entry.link), DROP_HOSTS) or not mentions(entry.title, ticker, tickers[ticker]):
            return
        if entry.published is None:
            undated += 1
            return
        if entry.published <= since:
            return
        item = NewsItem(
            id=item_id(ticker, entry),
            ticker=ticker,
            title=entry.title,
            link=entry.link,
            source=entry.source,
            published=entry.published,
            summary=entry.summary,
            kind=kind_of(entry, ticker, tickers[ticker]),
        )
        seen = found[ticker]
        title = item.title.lower()
        if item.id not in seen and all(i.title.lower() != title for i in seen.values()):
            seen[item.id] = item

    dead: list[str] = []
    for url in general:
        entries = _read(url, fetch)
        if entries is None:
            dead.append(url)
            continue
        for entry in entries:
            for ticker in tickers:
                add(ticker, entry)
    if per_ticker:
        failures = 0
        for ticker in tickers:
            url = per_ticker.format(ticker=ticker)
            # a feed that accepts connections and then hangs costs ATTEMPTS * TIMEOUT_S
            # per ticker; after a few in a row the rest are counted dead unread
            entries = _read(url, fetch) if failures < PER_TICKER_FAILURES else None
            if entries is None:
                failures += 1
                dead.append(url)
                continue
            failures = 0
            for entry in entries:
                add(ticker, entry)
    if dead:
        notes.append(f"{len(dead)} news feed(s) unavailable")
    if undated:
        log.info("left out %d undated entries", undated)

    def ranked(seen: dict[str, NewsItem]) -> tuple[NewsItem, ...]:
        newest = sorted(seen.values(), key=lambda i: i.published, reverse=True)
        return tuple(sorted(newest, key=lambda i: KIND_RANK[i.kind])[:cap])  # stable sort

    items = {t: ranked(seen) for t, seen in found.items()}
    return News(items=items, notes=tuple(notes))


def main(argv: list[str] | None = None) -> int:
    from tenq.config import load_watchlist

    parser = argparse.ArgumentParser(description="Print recent headlines for the watchlist")
    parser.add_argument("tickers", nargs="*", help="default: the whole watchlist")
    parser.add_argument("--hours", type=float, default=24.0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    watchlist = load_watchlist()
    tickers = [t.upper() for t in args.tickers] or list(watchlist.tickers)
    since = datetime.now(UTC) - timedelta(hours=args.hours)
    news = collect({t: watchlist.names.get(t) for t in tickers}, since)
    for note in news.notes:
        print(f"note: {note}")
    for ticker in tickers:
        print(f"{ticker}: {len(news.items[ticker])}")
        for item in news.items[ticker]:
            print(f"  {item.published:%m-%d %H:%M} {item.kind:<15} {item.id} {item.title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
