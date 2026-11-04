"""News headlines for each watchlist stock, from RSS feeds, each with a stable id.

Two kinds of feed: Yahoo's per-ticker feed, and general business feeds (CNBC,
MarketWatch). Yahoo's feed for a stock also carries stories about its peers that
never name it, so every entry, from either kind, must name the stock by symbol or
company name in its title or summary,
with the rules ticker-sentinel learned the hard way (`src/sentinel/news/matching.py`,
MIT, same author): a symbol matches only in capitals, so "net income" is never NET;
a one or two letter symbol needs a cashtag, brackets or an exchange prefix, so the
"S" in "U.S." is never a stock; a company name matches without its legal suffix.

Headline text is data from the open web, never instructions. Nothing here raises: a
dead feed is skipped with a note.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import logging
import re
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlsplit, urlunsplit

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
MIN_NAME = 4
EXCHANGES = r"NYSE(?: American)?|NASDAQ|Nasdaq|AMEX"
LEGAL_SUFFIXES = frozenset(
    {"inc", "incorporated", "corp", "corporation", "holdings", "holding", "ltd", "limited"}
    | {"plc", "co", "company", "nv", "sa", "ag", "se"}
)
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "ticker": self.ticker,
            "title": self.title,
            "link": self.link,
            "source": self.source,
            "published": self.published.isoformat(),
            "summary": self.summary,
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

    parsed = feedparser.parse(raw)
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
    if len(ticker) <= SHORT_SYMBOL:
        return re.compile(rf"(?:\${symbol}\b|\({symbol}\)|\b(?:{EXCHANGES}):\s*{symbol}\b)")
    return re.compile(rf"(?<![\w$.-]){symbol}(?![\w-])|\${symbol}\b")


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


def link_key(entry: Entry) -> str:
    """Same story, same key: a link without its tracking query, else the title."""
    if entry.link:
        parts = urlsplit(entry.link)
        return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, "", ""))
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
            text = f"{entry.title} {entry.summary}"
            for ticker, name in tickers.items():
                if mentions(text, ticker, name):
                    add(ticker, entry)
    if per_ticker:
        for ticker in tickers:
            url = per_ticker.format(ticker=ticker)
            entries = _read(url, fetch)
            if entries is None:
                dead.append(url)
                continue
            for entry in entries:
                if mentions(f"{entry.title} {entry.summary}", ticker, tickers[ticker]):
                    add(ticker, entry)
    if dead:
        notes.append(f"{len(dead)} news feed(s) unavailable")
    if undated:
        log.info("left out %d undated entries", undated)
    items = {
        t: tuple(sorted(seen.values(), key=lambda i: i.published, reverse=True)[:cap])
        for t, seen in found.items()
    }
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
            print(f"  {item.published:%m-%d %H:%M} {item.id} {item.title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
