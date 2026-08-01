#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTION="${1:-smoke}"
LOCK="$ROOT/skillopt/skillopt.lock.json"
TASKS="$ROOT/skillopt/fa_research_workflow_tasks_v1.json"
TARGET=".agents/skills/fa-research-workflow/SKILL.md"
SKILLOPT_URL="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["repository"])' "$LOCK")"
SKILLOPT_COMMIT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["commit"])' "$LOCK")"
SKILLOPT_REPO="${SKILLOPT_REPO:-$ROOT/.cache/skillopt}"

python3 "$ROOT/tools/validate_fa_skillopt.py" --project "$ROOT"

bootstrap() {
  if [[ ! -d "$SKILLOPT_REPO/.git" ]]; then
    git clone --filter=blob:none "$SKILLOPT_URL" "$SKILLOPT_REPO"
  fi
  if ! git -C "$SKILLOPT_REPO" cat-file -e "$SKILLOPT_COMMIT^{commit}" 2>/dev/null; then
    git -C "$SKILLOPT_REPO" fetch --quiet origin "$SKILLOPT_COMMIT"
  fi
  git -C "$SKILLOPT_REPO" checkout --quiet --detach "$SKILLOPT_COMMIT"
  local actual
  actual="$(git -C "$SKILLOPT_REPO" rev-parse --short=7 HEAD)"
  if [[ "$actual" != "$SKILLOPT_COMMIT" ]]; then
    echo "SkillOpt pin mismatch: expected $SKILLOPT_COMMIT, got $actual" >&2
    exit 1
  fi
}

run_sleep() {
  PYTHONPATH="$SKILLOPT_REPO${PYTHONPATH:+:$PYTHONPATH}" \
    python3 -m skillopt_sleep "$@"
}

codex_preflight() {
  local output
  output="$(mktemp)"
  trap 'rm -f "$output"' RETURN
  if ! codex exec --skip-git-repo-check --color never --sandbox read-only \
      -C "$ROOT" -o "$output" -- 'Reply with exactly: SKILLOPT_READY' \
      >/dev/null 2>&1; then
    echo "Codex backend preflight failed; refusing to record a misleading zero-score run." >&2
    return 1
  fi
  if [[ "$(tr -d '\r\n' < "$output")" != "SKILLOPT_READY" ]]; then
    echo "Codex backend preflight returned an unexpected response." >&2
    return 1
  fi
}

PREFERENCES="Only propose edits to the target skill. Never edit project memory, protocols, datasets, thresholds, results, or claims. Preserve every research boundary already present."

case "$ACTION" in
  validate)
    ;;
  bootstrap)
    bootstrap
    ;;
  status)
    bootstrap
    run_sleep status --project "$ROOT" --target-skill-path "$TARGET"
    ;;
  smoke)
    bootstrap
    run_sleep dry-run --project "$ROOT" --backend mock \
      --tasks-file "$TASKS" --target-skill-path "$TARGET" --json
    ;;
  dry-run)
    bootstrap
    codex_preflight
    run_sleep dry-run --project "$ROOT" --backend codex \
      --tasks-file "$TASKS" --target-skill-path "$TARGET" \
      --edit-budget 3 --max-tasks 9 --preferences "$PREFERENCES" --progress --json
    ;;
  run)
    bootstrap
    codex_preflight
    run_sleep run --project "$ROOT" --backend codex \
      --tasks-file "$TASKS" --target-skill-path "$TARGET" \
      --edit-budget 3 --max-tasks 9 --preferences "$PREFERENCES" --progress --json
    ;;
  adopt)
    if [[ "${FA_SKILLOPT_APPROVE_ADOPT:-}" != "YES" ]]; then
      echo "Refusing adoption. Review the staged report, then set FA_SKILLOPT_APPROVE_ADOPT=YES." >&2
      exit 2
    fi
    bootstrap
    staging="$(find "$ROOT/.skillopt-sleep/staging" -mindepth 1 -maxdepth 1 -type d -print 2>/dev/null | sort | tail -1)"
    if [[ -z "$staging" ]]; then
      echo "No staged SkillOpt proposal found." >&2
      exit 2
    fi
    python3 "$ROOT/tools/validate_fa_skillopt.py" --project "$ROOT" --staging "$staging"
    run_sleep adopt --project "$ROOT"
    ;;
  *)
    echo "usage: $0 {validate|bootstrap|status|smoke|dry-run|run|adopt}" >&2
    exit 2
    ;;
esac
