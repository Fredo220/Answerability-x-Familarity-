import hashlib
import json
from pathlib import Path


def test_registration_freezes_confirmatory_contract():
    text = Path("docs/rlmf_preregistration.md").read_text()
    for required in (
        "Date frozen: 2026-07-13",
        "a087e7a1e49f52aaa701add19cd80699b709fdef",
        "delta_cMFG_star >= 0.03",
        "did_gain >= 0.02",
        "No jailbreak claim",
    ):
        assert required in text


def test_vendored_reference_matches_upstream_manifest():
    upstream = json.loads(Path("third_party/rlmf/UPSTREAM.json").read_text())
    for name, digest in upstream["files"].items():
        payload = Path("third_party/rlmf", name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == digest
    assert upstream["commit"] == "a087e7a1e49f52aaa701add19cd80699b709fdef"


def test_vendored_trl_base_matches_manifest():
    upstream = json.loads(Path("third_party/trl/UPSTREAM.json").read_text())
    payload = Path("third_party/trl/grpo_trainer.py").read_bytes()
    assert hashlib.sha256(payload).hexdigest() == upstream["sha256"]
    assert upstream["tag"] == "v0.23.0"


def test_metacognition_prompt_is_bound_to_upstream_source():
    upstream = json.loads(Path("third_party/rlmf/UPSTREAM.json").read_text())
    prompt = Path("third_party/rlmf/metacognition_prompt.txt").read_bytes()
    assert hashlib.sha256(prompt).hexdigest() == upstream["metacognition_prompt_sha256"]
    assert upstream["metacognition_prompt_source"]
