#!/usr/bin/env bash
# Start the local SearXNG container, generating its secret on first run.
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -f .env ]]; then
  printf 'SEARXNG_SECRET=%s\n' "$(openssl rand -hex 32)" > .env
fi
docker compose up -d
for _ in $(seq 1 30); do
  if curl -fs "http://127.0.0.1:8888/search?q=test&format=json" -o /dev/null; then
    echo "SearXNG is up. Put SEARXNG_URL=http://127.0.0.1:8888 in the repo's .env."
    exit 0
  fi
  sleep 1
done
echo "SearXNG did not answer on 127.0.0.1:8888 within 30s; see: docker compose logs" >&2
exit 1
