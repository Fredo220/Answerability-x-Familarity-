#!/usr/bin/env python3
"""Freeze Source-v6 construction-validation identities before screening."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trajectory_extractor.fa_config import FAConfig
from trajectory_extractor.fa_development_screening import (
    _current_git_commit,
    _verify_clean_checkout,
    write_instrument_freeze_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--development-run-dir", type=Path, required=True)
    parser.add_argument("--success-criteria", type=Path, required=True)
    parser.add_argument("--manual-audit-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-commit")
    args = parser.parse_args()
    criteria = json.loads(args.success_criteria.read_text(encoding="utf-8"))
    if not isinstance(criteria, dict):
        raise ValueError("success criteria must be a JSON object")
    resolved_commit = args.git_commit or _current_git_commit()
    _verify_clean_checkout(resolved_commit)
    result = write_instrument_freeze_manifest(
        args.output,
        source_root=args.source_root,
        development_run_dir=args.development_run_dir,
        config=FAConfig.from_json(args.config),
        success_criteria=criteria,
        manual_audit_manifest=args.manual_audit_manifest,
        git_commit=resolved_commit,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
