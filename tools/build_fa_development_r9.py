#!/usr/bin/env python3
"""Build the frozen audit-qualified Source-v6 R9 development corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trajectory_extractor.fa_config import FAConfig
from trajectory_extractor.fa_development_r9 import derive_r9_source
from trajectory_extractor.fa_runtime import load_pinned_tokenizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--r8-root", type=Path, required=True)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = FAConfig.from_json(args.config)
    tokenizer = load_pinned_tokenizer(config)
    result = derive_r9_source(
        r8_root=args.r8_root,
        corrections_path=args.corrections,
        output_dir=args.output_dir,
        tokenizer=tokenizer,
        config=config,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
