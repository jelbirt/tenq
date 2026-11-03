# tenq

Cited what-changed briefs from SEC filings.

When a company on a 20-name software watchlist files a 10-Q or 10-K, tenq emails a short brief
of what changed and why. Every reason cites the filing passage it came from, and every number is
checked in code against the filing's tagged XBRL data before the email goes out. If the model
call fails, the email still goes out with the numbers only.

Status: slice 1 in progress (one company, one email, on GitHub Actions, traced).

## Development

```bash
uv sync
scripts/checks.sh
```

Briefs summarise public filings and are not investment advice.

## License

MIT. See [LICENSE](LICENSE).
