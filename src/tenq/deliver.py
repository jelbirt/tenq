"""The brief email: rendered by code from checked claims and tagged numbers, sent over SMTP.

`send` is adapted from ticker-sentinel's src/sentinel/deliver.py (MIT, same
author). Recipients and credentials come only from env vars.
"""

from __future__ import annotations

import html
import logging
import os
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage

from tenq.checker import CheckResult
from tenq.filings import FilingMeta, FilingStore
from tenq.metrics import MetricRow, MetricsTable

log = logging.getLogger(__name__)

DISCLAIMER = (
    "This brief summarises public SEC filings. It is not investment advice. "
    "Numbers are from the filing's tagged XBRL data and were checked by code; "
    "each reason cites the filing passage it came from."
)
QUOTE_CHARS = 280
TABLE_ORDER = (
    "revenue",
    "gross_margin",
    "operating_income",
    "operating_margin",
    "net_income",
    "operating_cash_flow",
    "free_cash_flow",
    "deferred_revenue",
    "cash",
)


@dataclass(frozen=True)
class Email:
    subject: str
    text: str
    html: str


def _usd(v: float | None, sign: bool = False) -> str:
    if v is None:
        return "n/a"
    a = abs(v)
    body = f"${a / 1e9:.2f}B" if a >= 1e9 else f"${a / 1e6:.1f}M" if a >= 1e6 else f"${a:,.0f}"
    return ("-" if v < 0 else "+" if sign else "") + body


def _pct(v: float | None, sign: bool = True) -> str:
    return "n/a" if v is None else f"{v * 100:+.1f}%" if sign else f"{v * 100:.1f}%"


def _pts(v: float | None) -> str:
    return "n/a" if v is None else f"{v * 100:+.1f} pts"


def _cells(row: MetricRow) -> list[str]:
    if row.unit == "ratio":
        vals = [_pct(row.number(f), sign=False) for f in ("current", "prior_quarter", "year_ago")]
        return [*vals, _pts(row.qoq_change), _pts(row.yoy_change)]
    vals = [_usd(row.number(f)) for f in ("current", "prior_quarter", "year_ago")]
    return [*vals, _pct(row.qoq_pct), _pct(row.yoy_pct)]


def _rows(table: MetricsTable) -> list[tuple[str, list[str]]]:
    return [
        (table.rows[name].label, _cells(table.rows[name]))
        for name in TABLE_ORDER
        if name in table.rows and table.rows[name].current is not None
    ]


def subject_for(meta: FilingMeta, table: MetricsTable) -> str:
    rev = table.rows.get("revenue")
    tail = f": revenue {_pct(rev.yoy_pct)} YoY" if rev and rev.yoy_pct is not None else ""
    return f"tenq: {meta.ticker} {meta.form} for the quarter ended {meta.period}{tail}"


def render(
    meta: FilingMeta,
    table: MetricsTable,
    result: CheckResult,
    store: FilingStore,
    *,
    summary_error: str | None = None,
    run_id: str = "",
    prompt_version: str = "",
) -> Email:
    """The email for one filing. With no kept claims it is numbers-only."""
    header = ["", "This quarter", "Prior quarter", "Year ago", "QoQ", "YoY"]
    rows = _rows(table)
    claims = []
    for c in result.kept:
        p = store.passage(c.passage_id)
        quote = " ".join(p.text.split())[:QUOTE_CHARS] if p else ""
        claims.append((c.claim, c.passage_id, quote))
    if summary_error is not None:
        note = (
            "The written summary was not available for this filing, so this email has numbers only."
        )
    elif not claims:
        note = "No reason passed the citation and number checks, so this email has numbers only."
    else:
        note = ""
    dropped = (
        f"{len(result.dropped)} claim(s) failed the checks and were left out."
        if result.dropped
        else ""
    )
    footer = f"Run {run_id}, prompt {prompt_version}." if run_id else ""

    # plain text
    t = [f"{meta.company} ({meta.ticker}) {meta.form}, quarter ended {meta.period}", ""]
    if claims:
        t.append("What changed and why")
        for claim, pid, quote in claims:
            t += [f"- {claim}", f'  Source {pid}: "{quote}"']
        t.append("")
    if note:
        t += [note, ""]
    t.append("Key numbers (" + ", ".join(header[1:]) + ")")
    t += [f"- {label}: " + " | ".join(cells) for label, cells in rows]
    t += ["", f"Filing: {meta.url}"]
    t += [x for x in (dropped, DISCLAIMER, footer) if x]

    # HTML: every value escaped; model text is untrusted
    e = html.escape
    h = [
        "<div style='font-family:Arial,sans-serif;max-width:720px'>",
        f"<h2 style='margin-bottom:4px'>{e(meta.company)} ({e(meta.ticker)}) {e(meta.form)}</h2>",
        f"<div style='color:#555'>Quarter ended {e(meta.period)}, filed {e(meta.filed)}</div>",
    ]
    if claims:
        h.append("<h3>What changed and why</h3><ul>")
        for claim, pid, quote in claims:
            h.append(
                f"<li style='margin-bottom:10px'>{e(claim)}<br>"
                f"<span style='color:#666;font-size:12px'>Source {e(pid)}: "
                f"&ldquo;{e(quote)}&rdquo;</span></li>"
            )
        h.append("</ul>")
    if note:
        h.append(f"<p><em>{e(note)}</em></p>")
    h.append("<h3>Key numbers</h3><table style='border-collapse:collapse;font-size:13px'>")
    h.append(
        "<tr>"
        + "".join(f"<th style='text-align:right;padding:4px 8px'>{e(x)}</th>" for x in header)
        + "</tr>"
    )
    for label, cells in rows:
        h.append(
            f"<tr><td style='padding:4px 8px'>{e(label)}</td>"
            + "".join(f"<td style='text-align:right;padding:4px 8px'>{e(c)}</td>" for c in cells)
            + "</tr>"
        )
    h.append("</table>")
    h.append(f"<p><a href='{e(meta.url, quote=True)}'>The filing on SEC EDGAR</a></p>")
    for x in (dropped, DISCLAIMER, footer):
        if x:
            h.append(f"<p style='color:#777;font-size:11px'>{e(x)}</p>")
    h.append("</div>")
    return Email(subject_for(meta, table), "\n".join(t) + "\n", "\n".join(h))


def send(email: Email) -> tuple[bool, str]:
    """Send to RECIPIENT_EMAILS. Returns (sent, status); never raises."""
    # `or` (not .get defaults): unset secrets reach Actions steps as empty strings
    user = os.environ.get("SMTP_USER") or None
    password = os.environ.get("SMTP_PASS") or None
    recipients = [r.strip() for r in os.environ.get("RECIPIENT_EMAILS", "").split(",") if r.strip()]
    if not user or not password or not recipients:
        return False, "email skipped: SMTP_USER / SMTP_PASS / RECIPIENT_EMAILS not all set"
    try:
        host = os.environ.get("SMTP_HOST") or "smtp.gmail.com"
        port = int(os.environ.get("SMTP_PORT") or "587")
        msg = EmailMessage()
        msg["Subject"] = email.subject
        msg["From"] = user
        msg["To"] = ", ".join(recipients)
        msg.set_content(email.text)
        msg.add_alternative(email.html, subtype="html")
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=60) as smtp:
                smtp.login(user, password)
                smtp.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=60) as smtp:
                smtp.starttls()
                smtp.login(user, password)
                smtp.send_message(msg)
    except Exception as exc:
        # the exception text is logged, never the credentials
        log.warning("email delivery failed: %s", type(exc).__name__)
        return False, f"email delivery failed: {type(exc).__name__}"
    return True, f"email sent to {len(recipients)} recipient(s)"
