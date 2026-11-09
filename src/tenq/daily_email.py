"""The daily email, rendered by code from the day's record (see tenq.daily).

It is rendered only from the record, so a resent email says exactly what was
recorded. At the top, one row per stock shows Jev's call, Opus's call and
whether they agree. Under that, each stock gets its checked reasons with
links to what they cite, then the balance lines and the disclaimer.

Every reason shown passed tenq.reasons, so every figure in it was written by
code. A link is shown only for an http(s) address: feed links are untrusted.
"""

from __future__ import annotations

import html
from typing import Any
from urllib.parse import urlsplit

from tenq.deliver import Email

DISCLAIMER = (
    "Not investment advice. Two methods each make a call on each stock the day's "
    "screen picked, from the same public facts: Jev answers questions and fixed "
    "rules turn the answers into a call; Opus makes its own. This is an experiment, "
    "and both are scored against prices later. Every figure in a reason was written "
    "by code from the facts sheet."
)


def _link(url: Any) -> str | None:
    if not isinstance(url, str):
        return None
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    return url if parts.scheme in ("http", "https") and parts.netloc else None


def sources(pack: dict[str, Any] | None) -> dict[str, tuple[str, str | None]]:
    """What each citable id points to: a short label and a link, if any."""
    if not pack:
        return {}
    out: dict[str, tuple[str, str | None]] = {}
    for h in pack.get("news") or []:
        site = f" ({h['site']})" if h.get("site") else ""
        out[h["id"]] = (f"{h.get('title', 'headline')}{site}", _link(h.get("link")))
    for r in pack.get("reports") or []:
        for e in r.get("excerpts") or []:
            label = f"{r.get('form') or '8-K'} filed {r.get('filed', '')}".strip()
            out[e["id"]] = (label, _link(r.get("url")))
    filing = pack.get("filing")
    if filing:
        label = f"the {filing.get('form', 'filing')} for the quarter ended {filing.get('period')}"
        link = _link(filing.get("url"))
        for n in pack.get("numbers") or []:
            if str(n.get("id", "")).startswith("fin:"):
                out[n["id"]] = (label, link)
    return out


def _cited(
    cites: list[str], where: dict[str, tuple[str, str | None]]
) -> list[tuple[str, str | None]]:
    seen: dict[tuple[str, str | None], None] = {}
    for c in cites:
        if c in where:
            seen[where[c]] = None
    return list(seen)


def _conf(v: Any) -> str:
    return f"{v:.0%}" if isinstance(v, int | float) else "confidence unreadable"


def _lean(opus: dict[str, Any]) -> str:
    """Opus's lean after its confidence; nothing for records made before it had one."""
    v = opus.get("lean")
    if isinstance(v, int | float) and not isinstance(v, bool):
        return f", lean {v:+.2f}"
    return f", lean not kept ({opus['lean_error']})" if opus.get("lean_error") else ""


def _jev_cell(stock: dict[str, Any]) -> str:
    call = stock.get("call")
    if call:
        return f"{call['call']} {_conf(call['confidence'])}"
    return "no call"


def _opus_cell(stock: dict[str, Any]) -> str:
    opus = stock.get("opus") or {}
    if opus.get("call"):
        return f"{opus['call']} {_conf(opus.get('confidence'))}"
    return "no call"


def _agree(stock: dict[str, Any]) -> str:
    jev = (stock.get("call") or {}).get("call")
    own = (stock.get("opus") or {}).get("call")
    if not jev or not own:
        return ""
    return "agree" if jev == own else "differ"


def subject_for(record: dict[str, Any]) -> str:
    stocks = record["stocks"]
    both = [s for s in stocks if _agree(s)]
    agree = sum(1 for s in both if _agree(s) == "agree")
    what = "daily check" if record.get("check_id") else "daily"
    weekly = "Weekly scorecard + " if (record.get("scorecard") or {}).get("lines") else ""
    failed = "SCREEN FAILED + " if screen_failed(record) else ""
    return (
        f"{failed}{weekly}tenq {what}, session {record['day']}: "
        f"the two calls agree on {agree} of {len(both)}"
    )


def screen_failed(record: dict[str, Any]) -> bool:
    return (record.get("screen") or {}).get("used") == "watchlist"


def screen_lines(record: dict[str, Any]) -> list[str]:
    """Which stocks were asked and why; empty for a check on named stocks."""
    sc = record.get("screen")
    if not sc:
        return []
    if sc.get("used") == "watchlist":
        why = "; ".join(sc.get("refused") or []) or "no reason given"
        return [
            f"SCREEN FAILED: {why}.",
            f"The {len(record['stocks'])} watchlist stocks were asked instead. "
            "Today's calls are not a screened day.",
        ]
    chosen, filled = sc.get("chosen") or [], sc.get("filled") or []
    out = [
        f"Screen {sc.get('version')}: {len(chosen)} of {sc.get('universe')} stocks passed"
        + (f", {len(filled)} more drawn at random as a check" if filled else "")
        + f"; {len(sc.get('held') or [])} Hold by screen"
        + (f"; {len(sc['unscreened'])} could not be screened" if sc.get("unscreened") else "")
        + "."
    ]
    out += [f"  {r['ticker']}: {'; '.join(r.get('reasons') or [])}" for r in chosen]
    if filled:
        out.append("  Random fill: " + ", ".join(filled))
    if sc.get("over_cap"):
        out.append("  Cut by the cap: " + ", ".join(r["ticker"] for r in sc["over_cap"]))
    return out


def scorecard_lines(record: dict[str, Any]) -> list[str]:
    """The week's scorecard for the top of the email, or why it is missing."""
    card = record.get("scorecard")
    if not card:
        return []
    if card.get("error"):
        return [f"Weekly scorecard: {card['error']}."]
    return [str(x) for x in card.get("lines") or []]


def _notes(stock: dict[str, Any]) -> list[str]:
    """What is missing from this stock's calls, in plain words."""
    out = [f"Note: {stock['note']}."] if stock.get("note") else []
    if stock.get("error"):
        return [f"No calls today: {stock['error']}."]
    jev = stock.get("jev") or {}
    if "call" not in stock:
        out.append(f"Jev made no call ({jev.get('error') or 'no usable answers'}).")
    opus = stock.get("opus") or {}
    if not opus.get("call"):
        out.append(f"Opus made no call ({opus.get('error') or 'no answer'}).")
    why = stock.get("why") or {}
    if "call" in stock and why.get("error"):
        out.append(f"No why for Jev's call ({why['error']}).")
    dropped = len(opus.get("dropped") or []) + len(why.get("dropped") or [])
    if dropped:
        out.append(f"{dropped} reason(s) failed the checks and were left out.")
    return out


def _money(v: Any) -> str:
    return f"${v:,.2f}" if isinstance(v, int | float) else "n/a"


def render(record: dict[str, Any]) -> Email:
    stocks = record["stocks"]
    e = html.escape
    rows = [(s["ticker"], _jev_cell(s), _opus_cell(s), _agree(s)) for s in stocks]
    foot = (
        f"Run {record['run_id']}: questions {record['versions']['questions']}, "
        f"rules {record['versions']['rules']}, prompts {record['versions']['call_prompt']} "
        f"and {record['versions']['why_prompt']}."
    )

    card = scorecard_lines(record)
    screened = screen_lines(record)
    failed = screen_failed(record)

    # plain text
    t = []
    if failed:
        t += [*screened, ""]
    if card:
        t += ["WEEKLY SCORECARD", *card, ""]
    t += [f"tenq daily calls, session {record['day']}", ""]
    if screened and not failed:
        t += [*screened, ""]
    t.append("Stock | Jev | Opus | the two")
    t += [" | ".join(r) for r in rows]
    t.append("")
    for s in stocks:
        where = sources(s.get("pack"))
        t.append(f"{s['ticker']} ({s['company']}), close {_money(s.get('close'))}")
        call = s.get("call")
        if call:
            t.append(f"  Jev: {call['call']}, confidence {_conf(call['confidence'])}")
            t += [f"    {line}" for line in call.get("explain") or []]
            kept = (s.get("why") or {}).get("kept") or []
            if kept:
                t.append("    Why (written by Opus for Jev's call, checked by code):")
                for r in kept:
                    t.append(f"    - {r['text']}")
                    for label, link in _cited(r["cites"], where):
                        t.append(f"      {label}" + (f": {link}" if link else ""))
        opus = s.get("opus") or {}
        if opus.get("call"):
            conf = _conf(opus.get("confidence"))
            t.append(f"  Opus: {opus['call']}, confidence {conf}{_lean(opus)}")
            for r in opus.get("kept") or []:
                t.append(f"    - {r['text']}")
                for label, link in _cited(r["cites"], where):
                    t.append(f"      {label}" + (f": {link}" if link else ""))
        t += [f"  {n}" for n in _notes(s)]
        t.append("")
    t += record.get("balance") or []
    t += ["", DISCLAIMER, foot]

    # HTML: every value escaped; model text and feed text are untrusted
    def reasons_html(items: list[dict[str, Any]], where: dict[str, tuple[str, str | None]]) -> str:
        parts = ["<ul style='margin:4px 0 8px 0'>"]
        for r in items:
            links = []
            for label, link in _cited(r["cites"], where):
                links.append(
                    f"<a href='{e(link, quote=True)}'>{e(label)}</a>" if link else e(label)
                )
            src = (
                f"<br><span style='color:#666;font-size:12px'>{'; '.join(links)}</span>"
                if links
                else ""
            )
            parts.append(f"<li style='margin-bottom:6px'>{e(r['text'])}{src}</li>")
        parts.append("</ul>")
        return "".join(parts)

    cell = "padding:4px 10px;text-align:left"
    h = ["<div style='font-family:Arial,sans-serif;max-width:760px'>"]
    if failed:
        h.append(
            "<div style='border:3px solid #b00;background:#fdecec;padding:10px 14px;"
            "margin-bottom:16px'>"
            f"<h2 style='margin:0 0 6px 0;color:#b00'>{e(screened[0])}</h2>"
            + "".join(f"<p style='font-size:14px;margin:2px 0'>{e(x)}</p>" for x in screened[1:])
            + "</div>"
        )
    if card:
        h.append(
            "<div style='border:2px solid #246;background:#eef3fa;padding:10px 14px;"
            "margin-bottom:16px'><h2 style='margin:0 0 6px 0'>Weekly scorecard</h2>"
            f"<p style='font-size:15px;margin:0 0 6px 0'><b>{e(card[0])}</b></p>"
            + "".join(f"<p style='font-size:13px;margin:2px 0'>{e(x)}</p>" for x in card[1:])
            + "</div>"
        )
    h += [
        f"<h2 style='margin-bottom:4px'>tenq daily calls, session {e(record['day'])}</h2>",
        *(
            [
                "<p style='font-size:13px;margin:4px 0 10px 0'>"
                + "<br>".join(e(x) for x in screened)
                + "</p>"
            ]
            if screened and not failed
            else []
        ),
        "<table style='border-collapse:collapse;font-size:13px'>",
        "<tr>"
        + "".join(f"<th style='{cell}'>{x}</th>" for x in ("Stock", "Jev", "Opus", ""))
        + "</tr>",
    ]
    for ticker, jev_cell, opus_cell, agree in rows:
        mark = "font-weight:bold;color:#a40" if agree == "differ" else "color:#777"
        h.append(
            f"<tr><td style='{cell}'>{e(ticker)}</td><td style='{cell}'>{e(jev_cell)}</td>"
            f"<td style='{cell}'>{e(opus_cell)}</td>"
            f"<td style='{cell};{mark}'>{e(agree)}</td></tr>"
        )
    h.append("</table>")
    for s in stocks:
        where = sources(s.get("pack"))
        h.append(
            f"<h3 style='margin:18px 0 4px 0'>{e(s['ticker'])} ({e(s['company'])}), "
            f"close {e(_money(s.get('close')))}</h3>"
        )
        call = s.get("call")
        if call:
            h.append(
                f"<div><b>Jev: {e(call['call'])}</b>, confidence {e(_conf(call['confidence']))}"
                "</div><div style='color:#555;font-size:12px'>"
                + "<br>".join(e(line) for line in call.get("explain") or [])
                + "</div>"
            )
            kept = (s.get("why") or {}).get("kept") or []
            if kept:
                h.append(
                    "<div style='margin-top:4px'>Why (written by Opus, checked by code):</div>"
                )
                h.append(reasons_html(kept, where))
        opus = s.get("opus") or {}
        if opus.get("call"):
            h.append(
                f"<div><b>Opus: {e(opus['call'])}</b>, confidence "
                f"{e(_conf(opus.get('confidence')) + _lean(opus))}</div>"
            )
            h.append(reasons_html(opus.get("kept") or [], where))
        for n in _notes(s):
            h.append(f"<div style='color:#777;font-size:12px'><em>{e(n)}</em></div>")
    for line in record.get("balance") or []:
        h.append(f"<p style='font-size:12px;margin:2px 0'>{e(line)}</p>")
    for x in (DISCLAIMER, foot):
        h.append(f"<p style='color:#777;font-size:11px'>{e(x)}</p>")
    h.append("</div>")
    return Email(subject_for(record), "\n".join(t) + "\n", "\n".join(h))
