# tenq

Cited what-changed briefs from SEC filings.

When a company on a 20-name software watchlist files a 10-Q or 10-K, tenq emails a short brief
of what changed and why. Every reason cites the filing passage it came from, and every number is
checked in code against the filing's tagged XBRL data before the email goes out. If the model
call fails, the email still goes out with the numbers only.

Status: slice 1 (one company, one email a filing, on GitHub Actions, traced).

## How one run works

A GitHub Actions workflow ([brief.yml](.github/workflows/brief.yml)) runs daily:

1. **Fetch** ([fetch.py](src/tenq/fetch.py), [edgar.py](src/tenq/edgar.py)): new 10-Q and 10-K
   filings for the active watchlist names come from SEC EDGAR through edgartools. Each is saved
   as normalised text split into passages with stable ids (`CRWD-0001535527-26-000031:23`, the
   filing and its first line), plus its tagged XBRL facts.
2. **Numbers** ([metrics.py](src/tenq/metrics.py)): revenue, deferred revenue, operating cash
   flow, free cash flow, margins and a few more for the quarter, with quarter-over-quarter and
   year-over-year changes computed in code. Quarters tagged only year-to-date are derived by
   difference, and restated periods take the newer filing's value.
3. **Search** ([search.py](src/tenq/search.py)): a throwaway local [xerj](https://github.com/xerj-org/xerj)
   node indexes the filings; queries are lexical (BM25) and hits map back to passage ids.
4. **Claude** ([llm.py](src/tenq/llm.py), [brief.py](src/tenq/brief.py)): `claude -p` runs with
   no built-in tools and only tenq's three read-only MCP tools
   ([mcp_server.py](src/tenq/mcp_server.py)): `get_metrics`, `search_filing`, `read_passage`.
   The prompt is versioned in [prompts/brief.md](prompts/brief.md). The answer is JSON claims,
   each with a passage id. The model never types a figure: it writes a number's id from
   `get_metrics` in curly brackets, `{fin:CRWD:revenue:yoy_pct}`.
5. **Check** ([checker.py](src/tenq/checker.py)): a claim is kept only if its passage exists in
   this filing or an earlier one, every id is one of the filing's numbers, and the model's own
   words hold no figure. Code then writes each number's clause from the tagged data
   ([reasons.py](src/tenq/reasons.py), the same code the daily calls use). Each id must stand
   as its own clause, so words cannot attach straight after code's number and change what it
   measures. Failures are dropped whole, never cut, and logged with a reason. What code does
   not judge is the meaning of the model's own words around a clause.
6. **Email** ([deliver.py](src/tenq/deliver.py)): the kept claims with their quoted passages and
   a table of the numbers, or the numbers alone if step 3 or 4 failed.
7. **Record and trace** ([run.py](src/tenq/run.py), [trace.py](src/tenq/trace.py)): a JSON run
   record is committed under [runs/](runs/), and a trace (steps, the model call with tokens and
   prompt version, each tool call) goes to Langfuse through its Python SDK. The record says the
   trace is stored only after a lookup through Langfuse's API finds it. The `trace-check`
   workflow sends one test trace the same way, to prove the secrets work.

## The daily calls

A second workflow ([daily.yml](.github/workflows/daily.yml), [daily.py](src/tenq/daily.py)) runs
on weekdays at 22:30 UTC (5:30pm New York in winter, 6:30pm in summer) and makes
two calls (Buy, Hold, Sell or Short) on every watchlist stock after the close, from the same
facts sheet ([pack.py](src/tenq/pack.py)): Jev answers text-only questions and fixed rules
([rules.py](src/tenq/rules.py), [config/rules.yaml](config/rules.yaml)) turn the answers into a
call; Opus makes its own call and writes the why for Jev's ([opus.py](src/tenq/opus.py)). Code
checks every reason and writes every figure in it ([reasons.py](src/tenq/reasons.py)). One email
shows both calls per stock with links to what each reason cites
([daily_email.py](src/tenq/daily_email.py)). The day's record, `runs/daily/<day>.json`, holds
the facts, both calls, the day's closes for reference and the time the calls were made. Each
stock gets its own Langfuse trace, grouped by day. `runs/` is written by both workflows, each
only its own files.

The email never goes out twice. The workflow makes the calls, then marks the record's email
as being sent and pushes that to main, and only then sends. If the push fails, nothing was
sent and the job can simply be rerun. A rerun that finds the day already emailed does nothing;
one that finds the email failed resends it from the record without asking the models again.

If a scheduled run fails before its email was sent, a short "tenq daily run failed" email links
to the run. A weekend or market holiday (the NYSE calendar from the `holidays` package) sends nothing; a
trading day whose closing price is not posted yet fails the run, so the notice goes out. A
surprise closure the package does not list yet sends one such notice; bumping `holidays` in the
lock fixes it.

When a run stops with "nothing run or sent", the day's record needs a person:
- **its email may have gone out** (the job stopped while sending, or the mail server's answer
  was lost): check the inbox. If the email arrived, set `"sent": true` in the record's `email`
  on main; if not, set `"sent": false`. Then dispatch again. Records are written on one line,
  so `python -m json.tool runs/daily/<day>.json` shows one readably first.
- **unfinished or unreadable**: it was never emailed. Delete it from main and dispatch again.

**How steady each method is.** The consistency workflow (manual only) gathers each stock's facts
once, then asks Jev and Opus for their calls several times (3 by default) on those same facts.
A method whose call changes on identical facts owes part of its future score to luck, so this
is the earliest read on either method, and worth repeating after any prompt, question or rules
change. It prints, per method, on how many stocks the call stayed the same and how far Jev's
score or Opus's confidence moved, and commits the record to `runs/consistency/`. Nothing is
emailed, and the scorecard never reads these calls. A full run is about 60 Claude runs, so
with Opus on it is refused on weekday evenings (20:00 to 02:00 UTC), when the daily run needs
the same subscription.

### The scorecard

Every call is scored by price alone ([scorecard.py](src/tenq/scorecard.py), settings in
[config/score.yaml](config/score.yaml)): did the stock beat or lag QQQ the way the call said,
from the first market open after the call was made to the close 20 sessions later. Buy counts
the stock's lead over QQQ, Short counts its lag, Sell counts half its lag and Hold counts
nothing. Prices come from one fresh fetch when a call is due, never from the closes stored in
the daily record, and each finished score goes into a permanent ledger,
`runs/daily/scores/<version>.json`. Changing any setting means a new version and a new ledger.

On the last session of each week the daily email opens with a scorecard. Its first line says
which method is ahead so far and by how much, from the first finished pair on, even when the
lead may be luck. The next line says how big a gap luck alone could give at this point (a 90%
range), or that the lead is settled. A lead can settle only after 13 weeks are scored, and only
when it is at least 2.4 standard errors wide: stricter than the one-look range, because the
card is read every week and each look is another chance for luck to pass; a lead between the
two gets a line saying so. A settled lead can fall back to not settled in a later week. The
range adds up each call's daily moves by calendar day, so stocks moving together on the same
day and calls held for weeks are each counted once. After that the card shows each method next
to simple baselines (always Hold, always Buy, buy last month's winners, and the same calls
shuffled across stocks), the days the two disagreed, how many calls are still open and the
week's cost. It also shows how well each method ordered the stocks each day: Jev by its rules
score, and Opus by its lean, a number from -1 to +1 it gives with each call (over the calls
that have one, on a day where at least 5 and at least 80% of its calls do; otherwise by call weight times confidence, and
the card says on how many days the lean was used). A lean that points against its own call is
dropped and the call stands. Only the full daily records count; check runs
never do. The scorecard never holds back the email: if it fails, the email says so and goes
out. `uv run python -m tenq.scorecard` prints the current card without saving anything: only
the daily workflow writes the ledger.

`uv run python -m tenq.replay` replays the baselines (always Hold, always Buy, last month's
winners) over the last 3 years of prices ([replay.py](src/tenq/replay.py)), scored exactly as a
live call is. Jev and Opus cannot be replayed, since the news they read is gone. This shows how
the baselines have done over the last 3 years and how much they swing from year to year. The
bar a live method must clear is still the baselines on the same days, on the scorecard. The
luck range here is for 3 years of calls; a few months of live calls have a much wider one. The
watchlist was chosen in 2026, which flatters every rule that mostly buys, always Buy most.

### The screen (preview only)

`uv run python -m tenq.screen` shows which stocks a wider daily run would send to Jev and Opus
([screen.py](src/tenq/screen.py), settings in [config/screen.yaml](config/screen.yaml)). It
asks no model anything. From the watchlist plus the Nasdaq-100
([config/universe.yaml](config/universe.yaml), 113 stocks) it passes a stock that filed an
8-K, 10-Q or 10-K (or a foreign company's 6-K, 20-F or 40-F) since the previous session, or
that moved against QQQ by more than 2.5 of its usual daily moves against QQQ over 1 or 5 days. At most 25 pass; filings come first, then the bigger move.
When fewer than 20 pass, the rest of the 20 are drawn at random from the stocks left out, the
same draw on a rerun, as a check on the screen itself. Everyone else would be "Hold by screen",
except a stock that could not be fully checked (prices behind the session, a gap in its prices,
too little history for both moves, or its EDGAR lookup failed): that is listed as not
screened, and is never drawn for the fill. The run stops, and its JSON says `"ok": false`, if
QQQ's bar for the session is missing, or more than 1 in 10 stocks' EDGAR lookups or prices
fail.
The manual `screen` workflow runs it with the EDGAR check on. The daily run does not use it yet.

## Guardrails

- The model never sees a shell, files or the web: `--tools ""`, `--strict-mcp-config`, an
  allowlist of three tools, no settings sources. A run that was offered or used any other tool
  is discarded. Filing text is data, never instructions.
- Every model failure (timeout, auth, bad output, wrong tools) falls back to a numbers-only email.
- The model's process gets its own token and no other secret; xerj gets none. Run records hold
  no secret, since they are public.
- A filing counts as briefed only once its email has gone out, and runs are serialized, so a
  filing is neither skipped nor emailed twice. Before the email goes out, its record (marked
  as sending) is saved to main; if that save fails, nothing is sent and the next run tries
  again. A record that is still marked as sending, says its email may have gone out, or cannot
  be read is never resent: the run skips it and goes red. To clear one, check the inbox and set
  `email.sent` in `runs/<TICKER>/<accession>.json` to `true` (it arrived) or `false` (send it
  again), or delete a record that cannot be read.
- SEC fair access: a contact email in the User-Agent and under 10 requests a second.

## Running it

```bash
uv sync
scripts/install-xerj.sh                        # pinned xerj release, checksum verified
EDGAR_IDENTITY="Your Name you@example.com" uv run python -m tenq.run --ticker CRWD --no-email
```

`--backfill N` also briefs the last N filings without a sent email; `--accession` briefs one
filing again. The workflow takes the same options through a manual dispatch. Secrets it uses:
`CLAUDE_CODE_OAUTH_TOKEN` (from `claude setup-token`), `EDGAR_IDENTITY`, `SMTP_USER`,
`SMTP_PASS`, `RECIPIENT_EMAILS`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`.

Version 0 runs on the owner's Claude subscription for personal and family use. All model calls
go through [llm.py](src/tenq/llm.py), so an API key can replace it.

## Development

```bash
scripts/checks.sh   # lint, format, types, tests: the bar CI runs
```

Tests never touch the network; they run on saved CRWD filings in
[tests/fixtures](tests/fixtures).

## Provenance

The `claude -p` call pattern and SMTP delivery are adapted from
[ticker-sentinel](https://github.com/jelbirt/ticker-sentinel) (MIT, same author).

Briefs summarise public filings and are not investment advice.

## License

MIT. See [LICENSE](LICENSE).
