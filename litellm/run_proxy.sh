#!/usr/bin/env bash
# Start the LiteLLM proxy for Sift on LITELLM_PORT (default 4000).
# Needs: pip install "litellm[proxy]"  (in its own venv is fine; set LITELLM_BIN)
set -euo pipefail
cd "$(dirname "$0")/.."

set -a; source .env; set +a      # OPENROUTER_API_KEY, OPENAI_*, LANGFUSE_*, LITELLM_*

# adept3o = Sift's existing OpenAI-compatible endpoint, unless overridden
export ADEPT3O_BASE_URL="${ADEPT3O_BASE_URL:-$OPENAI_BASE_URL}"
export ADEPT3O_API_KEY="${ADEPT3O_API_KEY:-${OPENAI_API_KEY:-none}}"
# LiteLLM's Langfuse callback reads LANGFUSE_HOST; Sift's .env calls it LANGFUSE_BASE_URL
export LANGFUSE_HOST="${LANGFUSE_HOST:-${LANGFUSE_BASE_URL:-}}"
# .env's DATABASE_URL is Sift's app database; LiteLLM would try to use it for its
# own (optional) key/spend database. The proxy runs without one.
unset DATABASE_URL
: "${LITELLM_MASTER_KEY:?set LITELLM_MASTER_KEY in .env (any secret starting with sk-)}"

exec "${LITELLM_BIN:-litellm}" --config litellm/config.yaml --port "${LITELLM_PORT:-4000}"
