import argparse
import json
from pathlib import Path

from trajectory_extractor import cli
from trajectory_extractor.fa_cli import dispatch_fa, register_fa_subcommands


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "familiarity_answerability_qwen06b_smoke.json"
)


def test_fa_commands_are_registered_with_explicit_config_and_root(tmp_path):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_fa_subcommands(subparsers)

    args = parser.parse_args(
        ["fa-build-pilot", "--config", str(CONFIG_PATH), "--root", str(tmp_path)]
    )

    assert args.command == "fa-build-pilot"
    assert args.config == str(CONFIG_PATH)
    assert args.root == str(tmp_path)


def test_fa_dispatch_is_isolated_and_cli_routes_fa_commands(tmp_path, capsys):
    args = argparse.Namespace(command="rlmf-prepare-data")
    assert dispatch_fa(args) is None

    exit_code = cli.main(
        ["fa-build-pilot", "--config", str(CONFIG_PATH), "--root", str(tmp_path)]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload == {"command": "fa-build-pilot", "status": "not_implemented"}
