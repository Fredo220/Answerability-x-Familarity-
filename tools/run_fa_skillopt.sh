#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTION="${1:-smoke}"
LOCK="$ROOT/skillopt/skillopt.lock.json"
SAFE_BASE="${TMPDIR:-/tmp}"
SAFE_ROOT=""
TARGET=".agents/skills/fa-research-workflow/SKILL.md"
TASKS="skillopt/fa_research_workflow_tasks_v1.json"
SKILLOPT_URL="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["repository"])' "$LOCK")"
SKILLOPT_COMMIT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["commit"])' "$LOCK")"
SKILLOPT_MODEL="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["codex_model"])' "$LOCK")"
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
  local actual dirty
  actual="$(git -C "$SKILLOPT_REPO" rev-parse HEAD)"
  dirty="$(git -C "$SKILLOPT_REPO" status --porcelain)"
  if [[ "$actual" != "$SKILLOPT_COMMIT" ]]; then
    echo "SkillOpt pin mismatch: expected $SKILLOPT_COMMIT, got $actual" >&2
    exit 1
  fi
  if [[ -n "$dirty" ]]; then
    echo "SkillOpt checkout is dirty; refusing to execute unpinned code." >&2
    exit 1
  fi
}

prepare_workspace() {
  SAFE_ROOT="$(mktemp -d "$SAFE_BASE/fa-skillopt.XXXXXX")"
  python3 "$ROOT/tools/prepare_fa_skillopt_workspace.py" \
    --project "$ROOT" --destination "$SAFE_ROOT" --allowed-root "$SAFE_BASE" \
    --codex-path "$(command -v codex)" >/dev/null
}

run_sleep() {
  PYTHONPATH="$SKILLOPT_REPO${PYTHONPATH:+:$PYTHONPATH}" \
    python3 -m skillopt_sleep "$@"
}

codex_preflight() {
  local output
  output="$(mktemp)"
  trap 'rm -f "$output"' RETURN
  if ! "$SAFE_ROOT/codex-sandboxed" exec --skip-git-repo-check --color never --sandbox read-only \
      -C "$SAFE_ROOT" -m "$SKILLOPT_MODEL" -o "$output" -- \
      'Reply with exactly: SKILLOPT_READY' \
      >/dev/null 2>&1; then
    echo "Codex backend preflight failed; refusing to record a misleading run." >&2
    return 1
  fi
  if [[ "$(tr -d '\r\n' < "$output")" != "SKILLOPT_READY" ]]; then
    echo "Codex backend preflight returned an unexpected response." >&2
    return 1
  fi
}

latest_staging() {
  find "$SAFE_ROOT/.skillopt-sleep/staging" -mindepth 1 -maxdepth 1 \
    -type d -exec stat -f '%m %N' {} \; 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-
}

PREFERENCES="Only propose edits to the target skill. Never edit project memory, protocols, datasets, thresholds, results, or claims. Preserve every research boundary already present."

case "$ACTION" in
  validate)
    ;;
  bootstrap)
    bootstrap
    ;;
  smoke)
    bootstrap
    prepare_workspace
    run_sleep dry-run --project "$SAFE_ROOT" --backend mock \
      --tasks-file "$SAFE_ROOT/$TASKS" --target-skill-path "$TARGET" --json
    ;;
  dry-run)
    bootstrap
    prepare_workspace
    codex_preflight
    run_sleep dry-run --project "$SAFE_ROOT" --backend codex \
      --model "$SKILLOPT_MODEL" --codex-path "$SAFE_ROOT/codex-sandboxed" \
      --tasks-file "$SAFE_ROOT/$TASKS" --target-skill-path "$TARGET" \
      --edit-budget 3 --max-tasks 9 --preferences "$PREFERENCES" --progress --json
    ;;
  run)
    bootstrap
    prepare_workspace
    codex_preflight
    run_sleep run --project "$SAFE_ROOT" --backend codex \
      --model "$SKILLOPT_MODEL" --codex-path "$SAFE_ROOT/codex-sandboxed" \
      --tasks-file "$SAFE_ROOT/$TASKS" --target-skill-path "$TARGET" \
      --edit-budget 3 --max-tasks 9 --preferences "$PREFERENCES" --progress --json
    staging="$(latest_staging)"
    [[ -n "$staging" ]] && echo "Review staged proposal: $staging"
    ;;
  evaluate-test)
    bootstrap
    staging="${2:-}"
    if [[ -z "$staging" || ! -d "$staging" ]]; then
      echo "usage: $0 evaluate-test /absolute/path/to/staging/run" >&2
      exit 2
    fi
    SAFE_ROOT="$(cd "$staging/../../.." && pwd)"
    codex_preflight
    python3 "$ROOT/tools/validate_fa_skillopt.py" --project "$ROOT" --staging "$staging"
    PYTHONPATH="$SKILLOPT_REPO${PYTHONPATH:+:$PYTHONPATH}" \
      python3 "$ROOT/tools/evaluate_fa_skillopt_test.py" \
        --workspace "$SAFE_ROOT" --staging "$staging" \
        --model "$SKILLOPT_MODEL" \
        --codex-path "$SAFE_ROOT/codex-sandboxed" \
        --expected-baseline-sha256 "$(shasum -a 256 "$ROOT/$TARGET" | awk '{print $1}')" \
        --expected-tasks-sha256 "$(shasum -a 256 "$ROOT/$TASKS" | awk '{print $1}')" \
        --skillopt-commit "$SKILLOPT_COMMIT" \
        --output "$staging/test_evaluation.json"
    echo "Proposal passed reserved tests. Apply it only through normal review and a repository commit."
    ;;
  adopt)
    echo "Automatic adoption is disabled. Review the staged diff and apply it manually." >&2
    exit 2
    ;;
  *)
    echo "usage: $0 {validate|bootstrap|smoke|dry-run|run|evaluate-test}" >&2
    exit 2
    ;;
esac
