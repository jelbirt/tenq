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
   each with a passage id and the numbers it uses.
5. **Check** ([checker.py](src/tenq/checker.py)): a claim is kept only if its passage exists in
   this filing or an earlier one, and every number it declares or writes matches the tagged data.
   Failures are dropped and logged with a reason.
6. **Email** ([deliver.py](src/tenq/deliver.py)): the kept claims with their quoted passages and
   a table of the numbers, or the numbers alone if step 3 or 4 failed.
7. **Record and trace** ([run.py](src/tenq/run.py), [trace.py](src/tenq/trace.py)): a JSON run
   record is committed under [runs/](runs/), and a trace (steps, the model call with tokens and
   prompt version, each tool call) goes to Langfuse over OpenTelemetry.

## Guardrails

- The model never sees a shell, files or the web: `--tools ""`, `--strict-mcp-config`, an
  allowlist of three tools, no settings sources. A run that was offered or used any other tool
  is discarded. Filing text is data, never instructions.
- Every model failure (timeout, auth, bad output, wrong tools) falls back to a numbers-only email.
- The model's process gets its own token and no other secret; xerj gets none. Run records hold
  no secret, since they are public.
- A filing counts as briefed only once its email has gone out, and runs are serialized, so a
  filing is neither skipped nor emailed twice.
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
