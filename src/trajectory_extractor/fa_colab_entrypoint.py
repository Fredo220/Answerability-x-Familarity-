"""Minimal Colab entrypoint that avoids importing unrelated study stacks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fa-colab")
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("fa-colab-preflight")
    preflight.add_argument("--root", default=".")
    preflight.add_argument("--lock", required=True)

    screening = commands.add_parser("fa-run-colab-screening")
    screening.add_argument("--config", required=True)
    screening.add_argument("--root", default=".")
    screening.add_argument("--checkpoint-root", required=True)
    screening.add_argument("--scratch-root", required=True)
    screening.add_argument("--git-commit", required=True)
    screening.add_argument("--bundle-path", required=True)
    screening.add_argument("--bundle-sha256", required=True)
    screening.add_argument("--launch-manifest")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "fa-colab-preflight":
            from trajectory_extractor.fa_colab_preflight import (
                run_colab_preflight,
            )

            payload = run_colab_preflight(Path(args.root), Path(args.lock))
        else:
            from trajectory_extractor.fa_cli import (
                _assemble_screened_matches,
                _run_screening,
                _screen_entities,
            )
            from trajectory_extractor.fa_colab_screening import (
                run_colab_screening,
            )
            from trajectory_extractor.fa_config import FAConfig

            root = Path(args.root)
            payload = run_colab_screening(
                FAConfig.from_json(args.config),
                root,
                args,
                run_screening=_run_screening,
                screen_entities=_screen_entities,
                assemble_screened_matches=_assemble_screened_matches,
            )
    except Exception as error:
        _emit(
            {
                "command": args.command,
                "error": {
                    "message": str(error),
                    "type": type(error).__name__,
                },
                "status": "error",
            }
        )
        return 3 if isinstance(error, (ImportError, OSError, RuntimeError)) else 2

    _emit({"command": args.command, **payload})
    return 3 if payload.get("status") == "infrastructure_failure" else 0


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
