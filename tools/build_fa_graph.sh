#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

exec uvx --from graphifyy==0.9.25 graphify extract . \
  --code-only \
  --max-workers "${GRAPHIFY_MAX_WORKERS:-2}" \
  --output . \
  "$@"
