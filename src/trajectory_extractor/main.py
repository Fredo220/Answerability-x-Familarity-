from __future__ import annotations

import argparse

from trajectory_extractor.fa_cli import dispatch_fa, register_fa_subcommands


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="feature-dynamics")
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_fa_subcommands(subparsers)

    args = parser.parse_args(argv)
    exit_code = dispatch_fa(args)
    if exit_code is None:
        parser.error(f"unsupported command: {args.command}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
