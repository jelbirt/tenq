#!/usr/bin/env bash
# Install a pinned xerj release (Apache-2.0, github.com/xerj-org/xerj) into
# ~/.local/bin, checking its sha256 first. Used by CI and the daily job.
# Linux x86_64 only; elsewhere install xerj yourself.
set -euo pipefail

VERSION="1.0.0-rc.100"
SHA256="25aec3e84b904c995bd2d58abe4f83badb8a6f9d1d56dbe2f22180bada9b00cc"
NAME="xerj-${VERSION}-x86_64-unknown-linux-gnu"
URL="https://github.com/xerj-org/xerj/releases/download/v${VERSION}/${NAME}.tar.gz"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
curl -fsSL --retry 3 -o "$tmp/xerj.tar.gz" "$URL"
echo "${SHA256}  $tmp/xerj.tar.gz" | sha256sum --check --quiet
tar -xzf "$tmp/xerj.tar.gz" -C "$tmp"
mkdir -p "$HOME/.local/bin"
install -m 0755 "$tmp/$NAME/xerj" "$HOME/.local/bin/xerj"
echo "installed xerj ${VERSION} to $HOME/.local/bin/xerj"
