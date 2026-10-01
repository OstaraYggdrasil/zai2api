#!/bin/bash
# zai2api persistent launcher.
# Requires ZAI_JWT (or ZAI_SESSION_TOKEN) in the environment.
# Browser profiles live under ./data/browser-profile (persistent disk).
set -euo pipefail
cd "$(dirname "$0")"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,::1}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,::1}"
export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-8000}"
export ZAI_TRANSPORT="${ZAI_TRANSPORT:-browser}"
export BROWSER_PROXY="${BROWSER_PROXY:-http://127.0.0.1:8899}"
: "${ZAI_JWT:?Set ZAI_JWT (or ZAI_SESSION_TOKEN) before starting}"
mkdir -p data/browser-profile
exec .venv/bin/python -m zai2api
