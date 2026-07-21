from __future__ import annotations

import argparse
import json


FA_COMMANDS = (
    "fa-screen-entities",
    "fa-build-pilot",
    "fa-build-confirmatory",
    "fa-audit-manifest",
    "fa-run-generation",
    "fa-score-behavior",
    "fa-extract-activations",
    "fa-fit-probes",
    "fa-seal-selection",
    "fa-unlock-endpoint",
    "fa-evaluate-behavior-test",
    "fa-evaluate-probe-test",
    "fa-evaluate-intervention-test",
    "fa-run-interventions",
    "fa-select-circuit-cases",
    "fa-audit-circuit-fidelity",
    "fa-build-report",
)


def register_fa_subcommands(subparsers: argparse._SubParsersAction) -> None:
    for command in FA_COMMANDS:
        parser = subparsers.add_parser(command)
        parser.add_argument("--config", required=True)
        parser.add_argument("--root", default=".")


def dispatch_fa(args: argparse.Namespace) -> int | None:
    if getattr(args, "command", None) not in FA_COMMANDS:
        return None
    print(json.dumps({"command": args.command, "status": "not_implemented"}))
    return 2
