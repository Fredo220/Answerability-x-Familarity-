from __future__ import annotations

import argparse
import builtins
import importlib
import json

import trajectory_extractor.fa_colab_entrypoint as entrypoint
import trajectory_extractor.fa_colab_preflight as preflight
import trajectory_extractor.fa_colab_screening as screening


def test_entrypoint_import_does_not_load_unrelated_study_dependencies(
    monkeypatch,
):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "datasets" or name.startswith("trajectory_extractor.rlmf"):
            raise AssertionError(f"unrelated import attempted: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    reloaded = importlib.reload(entrypoint)
    parser = reloaded._parser()

    assert isinstance(parser, argparse.ArgumentParser)


def test_preflight_command_emits_one_ready_payload(monkeypatch, capsys):
    monkeypatch.setattr(
        preflight,
        "run_colab_preflight",
        lambda root, lock: {
            "status": "ready",
            "lock_path": str(root / lock),
            "torch_version": "2.7.1",
        },
    )

    exit_code = entrypoint.main(
        [
            "fa-colab-preflight",
            "--root",
            "/content/repo",
            "--lock",
            "requirements/fa-core.lock",
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {
        "command": "fa-colab-preflight",
        "lock_path": "/content/repo/requirements/fa-core.lock",
        "status": "ready",
        "torch_version": "2.7.1",
    }


def test_screening_command_requires_and_forwards_frozen_identity(
    monkeypatch, capsys, tmp_path
):
    observed = {}

    def fake_run(config, root, args, **transactions):
        observed.update(
            {
                "config": config,
                "root": root,
                "args": args,
                "transactions": transactions,
            }
        )
        return {
            "status": "assembled",
            "count": 244,
            "manifest": str(root / "collection.manifest.json"),
        }

    monkeypatch.setattr(screening, "run_colab_screening", fake_run)
    bundle = tmp_path / "fa-study.bundle"
    bundle.write_bytes(b"bundle")

    exit_code = entrypoint.main(
        [
            "fa-run-colab-screening",
            "--config",
            "configs/familiarity_answerability_gemma2_2b.json",
            "--root",
            str(tmp_path),
            "--checkpoint-root",
            str(tmp_path / "checkpoints"),
            "--scratch-root",
            str(tmp_path / "scratch"),
            "--git-commit",
            "a" * 40,
            "--bundle-path",
            str(bundle),
            "--bundle-sha256",
            "b" * 64,
        ]
    )

    assert exit_code == 0
    assert observed["root"] == tmp_path
    assert observed["args"].checkpoint_root == str(tmp_path / "checkpoints")
    assert set(observed["transactions"]) == {
        "run_screening",
        "screen_entities",
        "assemble_screened_matches",
    }
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "fa-run-colab-screening"
    assert payload["status"] == "assembled"
    assert payload["count"] == 244
