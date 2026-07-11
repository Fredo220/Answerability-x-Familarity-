import json

import pytest

from trajectory_extractor import load_prompts


def test_load_prompts_reads_named_prompt_mapping(tmp_path):
    prompts_path = tmp_path / "prompts.json"
    prompts_path.write_text(
        json.dumps(
            {
                "Baseline State": "When was Albert Einstein born?",
                "Anomalous State": "Tell me about a fabricated discovery.",
            }
        )
    )

    prompts = load_prompts(prompts_path)

    assert prompts == {
        "Baseline State": "When was Albert Einstein born?",
        "Anomalous State": "Tell me about a fabricated discovery.",
    }


def test_load_prompts_rejects_empty_prompt_text(tmp_path):
    prompts_path = tmp_path / "prompts.json"
    prompts_path.write_text(json.dumps({"Baseline State": ""}))

    with pytest.raises(ValueError, match="non-empty"):
        load_prompts(prompts_path)


def test_load_prompts_requires_json_object(tmp_path):
    prompts_path = tmp_path / "prompts.json"
    prompts_path.write_text(json.dumps(["not", "a", "mapping"]))

    with pytest.raises(ValueError, match="JSON object"):
        load_prompts(prompts_path)
