from __future__ import annotations

import json
from pathlib import Path

import pytest

from trajectory_extractor.fa_config import FAConfig
from trajectory_extractor.fa_confirmatory_source import REGISTERED_DOMAINS, SourceRecord
from trajectory_extractor.fa_development_screening import (
    run_development_screening,
    write_instrument_freeze_manifest,
)
from trajectory_extractor.fa_development_source import (
    DevelopmentSourceDesign,
    assign_development_pools,
    materialize_development_manifests,
    write_development_source,
)


class FakeRunner:
    def __init__(self, answers: dict[str, str], *, fail_after: int | None = None):
        self.answers = answers
        self.fail_after = fail_after
        self.calls = 0
        self.rendered: list[str] = []

    def render_prompt(self, prompt: str) -> str:
        self.rendered.append(prompt)
        return f"<bos>{prompt}<assistant>"

    def generate(self, prompts, generation):
        assert generation == {
            "do_sample": False,
            "max_new_tokens": 16,
            "temperature": 0.0,
        }
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError("simulated interruption")
        self.calls += 1
        return [
            self.answers[prompt.removeprefix("<bos>").removesuffix("<assistant>")]
            for prompt in prompts
        ]


def _config() -> FAConfig:
    return FAConfig.from_json("configs/familiarity_answerability_gemma2_2b.json")


def _record(domain: str, index: int) -> SourceRecord:
    offset = REGISTERED_DOMAINS.index(domain) * 1_000
    return SourceRecord(
        qid=f"Q{50_000 + offset + index}",
        label=f"{domain.title()} {index}",
        domain=domain,
        sitelinks=100,
        source_rank=index,
        property_values=(
            ("P1", (f"answer-{domain}-{index}-1",)),
            ("P2", (f"answer-{domain}-{index}-2",)),
            ("P3", (f"answer-{domain}-{index}-3",)),
        ),
    )


def _source(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    design = DevelopmentSourceDesign(candidates_per_domain_per_split=1)
    records = {
        domain: (_record(domain, 1), _record(domain, 2))
        for domain in REGISTERED_DOMAINS
    }
    manifests = materialize_development_manifests(
        assign_development_pools(records, design),
        design=design,
        retrieval_date="2026-07-25",
        query_hashes={domain: domain * 4 for domain in REGISTERED_DOMAINS},
    )
    root = tmp_path / "source"
    write_development_source(root, manifests, design=design)
    answers = {
        question.prompt: question.accepted_aliases[0]
        for candidates, questions in manifests.values()
        for question in questions
    }
    return root, answers


def _freeze(tmp_path: Path, source: Path) -> Path:
    path = tmp_path / "instrument_freeze.json"
    write_instrument_freeze_manifest(
        path,
        source_root=source,
        config=_config(),
        success_criteria={
            "minimum_qualified_per_domain": 20,
            "minimum_reserve_factor": 1.5,
        },
        git_commit="a" * 40,
    )
    return path


def test_runner_generates_exactly_three_rendered_prompts_and_writes_yield(tmp_path):
    source, answers = _source(tmp_path)
    runner = FakeRunner(answers)

    result = run_development_screening(
        _config(),
        source,
        "instrument_development",
        tmp_path / "output",
        batch_size=2,
        runner=runner,
    )

    assert result["status"] == "completed"
    assert result["item_count"] == 12
    assert result["candidate_count"] == 4
    assert len(runner.rendered) == 12
    assert result["summary"]["qualified_count"] == 4
    rows = [
        json.loads(line)
        for line in Path(result["items_path"]).read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 12
    assert all(row["split"] == "instrument_development" for row in rows)
    assert all(row["kind"] == "development_screening_item" for row in rows)


@pytest.mark.parametrize(
    "split",
    ["mechanism_train", "behavior_test", "probe_test", "intervention_test", "other"],
)
def test_runner_rejects_non_development_splits_before_model_call(tmp_path, split):
    source, answers = _source(tmp_path)
    runner = FakeRunner(answers)

    with pytest.raises(ValueError, match="development split"):
        run_development_screening(
            _config(), source, split, tmp_path / "output", runner=runner
        )

    assert runner.calls == 0


def test_runner_rejects_wrong_model_identity_before_model_call(tmp_path):
    source, answers = _source(tmp_path)
    runner = FakeRunner(answers)
    config = _config()
    object.__setattr__(config, "model_id", "Qwen/Qwen3-0.6B")

    with pytest.raises(ValueError, match="pinned confirmatory"):
        run_development_screening(
            config,
            source,
            "instrument_development",
            tmp_path / "output",
            runner=runner,
        )

    assert runner.calls == 0


def test_runner_rejects_tampered_source_before_model_call(tmp_path):
    source, answers = _source(tmp_path)
    runner = FakeRunner(answers)
    candidate_path = source / "candidate_entities_instrument_development_v1.json"
    candidate_path.write_text(candidate_path.read_text() + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash"):
        run_development_screening(
            _config(),
            source,
            "instrument_development",
            tmp_path / "output",
            runner=runner,
        )

    assert runner.calls == 0


def test_runner_resumes_only_missing_immutable_batches(tmp_path):
    source, answers = _source(tmp_path)
    output = tmp_path / "output"
    freeze = _freeze(tmp_path, source)
    interrupted = FakeRunner(answers, fail_after=1)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_development_screening(
            _config(),
            source,
            "construction_validation",
            output,
            batch_size=3,
            runner=interrupted,
            freeze_manifest=freeze,
            git_commit="a" * 40,
        )

    resumed = FakeRunner(answers)
    result = run_development_screening(
        _config(),
        source,
        "construction_validation",
        output,
        batch_size=3,
        runner=resumed,
        freeze_manifest=freeze,
        git_commit="a" * 40,
    )

    assert result["status"] == "completed"
    assert result["resumed_batch_count"] == 1
    assert resumed.calls == 3


def test_construction_validation_requires_matching_freeze_before_model_call(tmp_path):
    source, answers = _source(tmp_path)
    runner = FakeRunner(answers)

    with pytest.raises(ValueError, match="frozen instrument"):
        run_development_screening(
            _config(),
            source,
            "construction_validation",
            tmp_path / "output",
            runner=runner,
            git_commit="a" * 40,
        )
    assert runner.calls == 0

    freeze = _freeze(tmp_path, source)
    payload = json.loads(freeze.read_text())
    payload["git_commit"] = "b" * 40
    freeze.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="freeze manifest identity"):
        run_development_screening(
            _config(),
            source,
            "construction_validation",
            tmp_path / "output",
            runner=runner,
            freeze_manifest=freeze,
            git_commit="a" * 40,
        )
    assert runner.calls == 0


def test_runner_fails_closed_if_checkpoint_manifest_is_tampered(tmp_path):
    source, answers = _source(tmp_path)
    output = tmp_path / "output"
    runner = FakeRunner(answers)
    run_development_screening(
        _config(),
        source,
        "instrument_development",
        output,
        batch_size=3,
        runner=runner,
    )
    manifest = next(output.rglob("batch-*.manifest.json"))
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["row_count"] = 999
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint"):
        run_development_screening(
            _config(),
            source,
            "instrument_development",
            output,
            batch_size=3,
            runner=FakeRunner(answers),
        )
