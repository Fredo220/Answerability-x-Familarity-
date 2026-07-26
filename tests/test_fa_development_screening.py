from __future__ import annotations

import hashlib
import inspect
import json
import shutil
from pathlib import Path

import pytest

import trajectory_extractor.fa_development_screening as development_screening
from trajectory_extractor.fa_config import FAConfig
from trajectory_extractor.fa_confirmatory_source import REGISTERED_DOMAINS, SourceRecord
from trajectory_extractor.fa_development_screening import (
    development_screening_parser_sha256,
    evaluate_instrument_readiness,
    parse_development_screening_answer,
    run_development_screening,
    write_instrument_freeze_manifest,
)
from trajectory_extractor.fa_development_source import (
    DEVELOPMENT_DOMAIN_FIELDS,
    DevelopmentSourceDesign,
    assign_development_pools,
    build_manual_error_audit_packet,
    compile_manual_error_audit,
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


def _criteria(
    *,
    candidate_count: int = 4,
    minimum_qualified_per_domain: int = 1,
    require_human_audit: bool = False,
) -> dict[str, object]:
    criteria: dict[str, object] = {
        "schema_version": 2,
        "source_revision": "fa-development-source-v6-r7",
        "development_gate": {
            "candidate_count": candidate_count,
            "prompt_count": candidate_count * 3,
            "qualification_threshold": 2,
            "minimum_qualified_per_domain": minimum_qualified_per_domain,
            "minimum_success_by_domain_relation": {
                domain: {
                    field.property_id: 1
                    for field in DEVELOPMENT_DOMAIN_FIELDS[domain]
                }
                for domain in REGISTERED_DOMAINS
            },
        },
    }
    if require_human_audit:
        criteria["human_audit"] = {
            "independent_initial_raters": 2,
            "adjudicator_required_on_disagreement": True,
            "required_before_construction_validation": True,
            "sample_per_domain": 1,
            "success_sample_per_domain": 1,
            "seed": 20260725,
            "disallowed_error_labels": [
                "ambiguous_ground_truth",
                "incomplete_alias_set",
                "wrong_granularity",
                "parser_failure",
                "source_error",
                "other",
            ],
            "maximum_disallowed_count": 0,
            "maximum_scoring_disagreement_count": 0,
        }
    return criteria


def _semantic_audit(tmp_path: Path, source: Path) -> Path:
    integrity_path = source / "source_integrity_v1.json"
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    path = tmp_path / "semantic_audit.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "fa_source_v6_pre_model_semantic_audit",
                "source_revision": integrity["source_revision"],
                "source_integrity_sha256": hashlib.sha256(
                    integrity_path.read_bytes()
                ).hexdigest(),
                "auditor_id": "independent-test-auditor",
                "status": "passed",
                "blocker_count": 0,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


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


def _source(
    tmp_path: Path,
    *,
    candidates_per_domain_per_split: int = 1,
) -> tuple[Path, dict[str, str]]:
    design = DevelopmentSourceDesign(
        candidates_per_domain_per_split=candidates_per_domain_per_split
    )
    records = {
        domain: tuple(
            _record(domain, index)
            for index in range(1, candidates_per_domain_per_split * 2 + 1)
        )
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


def _freeze(tmp_path: Path, source: Path, answers: dict[str, str]) -> Path:
    criteria = _criteria()
    development = run_development_screening(
        _config(),
        source,
        "instrument_development",
        tmp_path / "development-output",
        runner=FakeRunner(answers),
        success_criteria=criteria,
        pre_model_semantic_audit=_semantic_audit(tmp_path, source),
        git_commit="a" * 40,
    )
    path = tmp_path / "instrument_freeze.json"
    write_instrument_freeze_manifest(
        path,
        source_root=source,
        development_run_dir=Path(development["summary_path"]).parent,
        config=_config(),
        success_criteria=criteria,
        git_commit="a" * 40,
    )
    return path


def _manual_audit(
    path: Path,
    development: dict[str, object],
    *,
    error_label: str = "relation_unknown",
    seed: int = 20260725,
) -> Path:
    run_dir = Path(str(development["summary_path"])).parent
    items = [
        json.loads(line)
        for line in (run_dir / "screening_items.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    packet = build_manual_error_audit_packet(
        items,
        sample_per_domain=1,
        success_sample_per_domain=1,
        seed=20260725,
    )
    scored_by_question = {
        str(row["question_id"]): bool(row["is_correct"]) for row in items
    }
    ratings = [
        {
            "audit_id": row["audit_id"],
            "rater_id": rater,
            "round": 1,
            "error_label": (
                "no_error"
                if scored_by_question[str(row["question_id"])]
                else error_label
            ),
        }
        for row in packet
        for rater in ("rater-a", "rater-b")
    ]
    manifest = {
        "schema_version": 1,
        "kind": "fa_source_v6_manual_error_audit",
        "source_revision": "fa-development-source-v6-r7",
        "development_execution_identity_sha256": development[
            "execution_identity_sha256"
        ],
        "items_sha256": development["items_sha256"],
        "summary_sha256": development["summary_sha256"],
        "sample_per_domain": 1,
        "success_sample_per_domain": 1,
        "seed": seed,
        "packet": list(packet),
        "ratings": ratings,
        "compiled": compile_manual_error_audit(packet, ratings),
    }
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return path


def test_cli_forwards_explicit_git_commit_for_archive_runtime(
    tmp_path, monkeypatch, capsys
):
    seen: dict[str, object] = {}

    def fake_run(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return {"status": "completed"}

    monkeypatch.setattr(development_screening, "run_development_screening", fake_run)
    monkeypatch.setattr(
        development_screening,
        "_verify_clean_checkout",
        lambda commit: None,
    )
    commit = "c" * 40
    criteria_path = tmp_path / "criteria.json"
    criteria_path.write_text(json.dumps(_criteria()), encoding="utf-8")
    audit_path = tmp_path / "audit.json"
    audit_path.write_text("{}", encoding="utf-8")

    exit_code = development_screening.main(
        [
            "--config",
            "configs/familiarity_answerability_gemma2_2b.json",
            "--source-root",
            str(tmp_path / "source"),
            "--split",
            "instrument_development",
            "--output-root",
            str(tmp_path / "output"),
            "--success-criteria",
            str(criteria_path),
            "--pre-model-semantic-audit",
            str(audit_path),
            "--git-commit",
            commit,
        ]
    )

    assert exit_code == 0
    assert seen["kwargs"]["git_commit"] == commit
    assert seen["kwargs"]["success_criteria"] == _criteria()
    assert seen["kwargs"]["pre_model_semantic_audit"] == audit_path
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


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
        success_criteria=_criteria(),
        pre_model_semantic_audit=_semantic_audit(tmp_path, source),
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
    assert all(row["completion"] == row["parsed_completion"] for row in rows)


@pytest.mark.parametrize(
    ("raw_output", "aliases", "answer"),
    [
        ("<think>work</think>\nFrance", ("France",), "France"),
        ("Answer: New York", ("New York",), "New York"),
        ("Final answer: 'physics'", ("physics",), "physics"),
        ('"France"', ("France",), "France"),
        (
            "Indian Standard Time (IST)",
            ("Indian Standard Time", "IST"),
            "Indian Standard Time",
        ),
        ("UTC+05:30", ("UTC+05:30",), "UTC+05:30"),
    ],
)
def test_development_parser_accepts_only_registered_answer_wrappers(
    raw_output, aliases, answer
):
    assert parse_development_screening_answer(raw_output, aliases) == answer


@pytest.mark.parametrize(
    ("raw_output", "aliases"),
    [
        ("Paris, France", ("Paris",)),
        ("The answer is Paris", ("Paris",)),
        ("Finalist: Paris", ("Paris",)),
        ("France or Belgium", ("France", "Belgium")),
        ("Paris (Texas)", ("Paris",)),
    ],
)
def test_development_parser_does_not_create_alias_false_positives(
    raw_output, aliases
):
    assert parse_development_screening_answer(raw_output, aliases) == raw_output


def test_development_parser_hash_binds_exact_implementation():
    expected = development_screening._canonical_sha256(
        {
            "revision": "fa-development-screening-answer-v2",
            "implementation": inspect.getsource(
                development_screening.parse_development_screening_answer
            ),
            "rules": [
                "strip",
                "suffix-after-final-think-close",
                "last-nonempty-line",
                "registered-answer-prefix-only",
                "single-matching-quote-pair",
                "parenthetical-only-when-both-parts-are-registered-aliases",
            ],
        }
    )

    assert development_screening_parser_sha256() == expected


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


def test_instrument_screening_does_not_open_validation_manifest(tmp_path):
    source, answers = _source(tmp_path)
    validation_path = (
        source / "candidate_entities_construction_validation_v1.json"
    )
    validation_path.write_text("not opened during development", encoding="utf-8")

    result = run_development_screening(
        _config(),
        source,
        "instrument_development",
        tmp_path / "output",
        runner=FakeRunner(answers),
        success_criteria=_criteria(),
        pre_model_semantic_audit=_semantic_audit(tmp_path, source),
        git_commit="a" * 40,
    )

    assert result["status"] == "completed"


def test_ordered_prompts_rejects_orphan_questions():
    candidate = development_screening.CandidateEntity(
        entity_id="entity-1",
        qid="Q1",
        name="Entity",
        coarse_type="person",
        split="instrument_development",
        source_query="query",
        source_provenance="source",
        screening_aliases=(("A",), ("B",), ("C",)),
    )
    questions = tuple(
        development_screening.ScreeningQuestion(
            question_id=f"entity-1-q{index}",
            qid="Q1",
            prompt=f"Question {index}?",
            accepted_aliases=(alias,),
            source_provenance="source",
        )
        for index, alias in enumerate(("A", "B", "C"), start=1)
    )
    orphan = development_screening.ScreeningQuestion(
        question_id="orphan-q1",
        qid="Q2",
        prompt="Orphan?",
        accepted_aliases=("D",),
        source_provenance="source",
    )

    with pytest.raises(ValueError, match="orphan"):
        development_screening._ordered_prompts((candidate,), questions + (orphan,))


def test_runner_resumes_only_missing_immutable_batches(tmp_path):
    source, answers = _source(tmp_path)
    output = tmp_path / "output"
    freeze = _freeze(tmp_path, source, answers)
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


def test_runner_restores_verified_batches_from_external_checkpoint(tmp_path):
    source, answers = _source(tmp_path)
    output = tmp_path / "local-output"
    checkpoint_root = tmp_path / "drive-checkpoints"
    criteria = _criteria()

    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_development_screening(
            _config(),
            source,
            "instrument_development",
            output,
            batch_size=3,
            runner=FakeRunner(answers, fail_after=1),
            success_criteria=criteria,
            pre_model_semantic_audit=_semantic_audit(tmp_path, source),
            checkpoint_root=checkpoint_root,
            git_commit="a" * 40,
        )

    shutil.rmtree(output)
    resumed = FakeRunner(answers)
    result = run_development_screening(
        _config(),
        source,
        "instrument_development",
        output,
        batch_size=3,
        runner=resumed,
        success_criteria=criteria,
        pre_model_semantic_audit=_semantic_audit(tmp_path, source),
        checkpoint_root=checkpoint_root,
        git_commit="a" * 40,
    )

    assert result["resumed_batch_count"] == 1
    assert resumed.calls == 3
    assert Path(result["checkpoint_metadata"]).is_file()


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

    freeze = _freeze(tmp_path, source, answers)
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
        success_criteria=_criteria(),
        pre_model_semantic_audit=_semantic_audit(tmp_path, source),
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
            success_criteria=_criteria(),
            pre_model_semantic_audit=_semantic_audit(tmp_path, source),
        )


def test_instrument_readiness_gate_checks_domains_relations_and_place_slots():
    summary = {
        "entity_count": 4,
        "by_domain": {
            domain: {"qualified_count": 1} for domain in REGISTERED_DOMAINS
        },
    }
    items = [
        {
            "domain": domain,
            "question_id": f"development-{domain}-q{index}",
            "is_correct": True,
        }
        for domain in REGISTERED_DOMAINS
        for index in range(1, 4)
    ]
    criteria = {
        "schema_version": 2,
        "source_revision": "fa-development-source-v6-r5",
        "development_gate": {
            "candidate_count": 4,
            "prompt_count": 12,
            "qualification_threshold": 2,
            "minimum_qualified_per_domain": 1,
            "minimum_success_by_domain_relation": {
                domain: {
                    field.property_id: 1
                    for field in DEVELOPMENT_DOMAIN_FIELDS[domain]
                }
                for domain in REGISTERED_DOMAINS
            },
        }
    }

    passed = evaluate_instrument_readiness(summary, items, criteria)
    assert passed["gate_passed"]
    assert passed["observed"]["success_by_domain_relation"]["place"] == {
        "P17": 1,
        "P131": 1,
        "P30": 1,
    }

    items[3]["is_correct"] = False
    failed = evaluate_instrument_readiness(summary, items, criteria)
    assert not failed["gate_passed"]
    assert failed["failed_criteria"] == ["success_by_domain_relation"]


def test_relation_gate_does_not_pool_same_property_across_domains():
    summary = {
        "entity_count": 4,
        "by_domain": {
            domain: {"qualified_count": 1} for domain in REGISTERED_DOMAINS
        },
    }
    items = [
        {
            "domain": domain,
            "question_id": f"development-{domain}-q{index}",
            "is_correct": True,
        }
        for domain in REGISTERED_DOMAINS
        for index in range(1, 4)
    ]
    organization_p17 = next(
        item
        for item in items
        if item["domain"] == "organization"
        and item["question_id"].endswith("-q1")
    )
    organization_p17["is_correct"] = False
    criteria = {
        "schema_version": 2,
        "source_revision": "fa-development-source-v6-r5",
        "development_gate": {
            "candidate_count": 4,
            "prompt_count": 12,
            "qualification_threshold": 2,
            "minimum_qualified_per_domain": 1,
            "minimum_success_by_domain_relation": {
                domain: {
                    field.property_id: 1
                    for field in DEVELOPMENT_DOMAIN_FIELDS[domain]
                }
                for domain in REGISTERED_DOMAINS
            },
        },
    }

    result = evaluate_instrument_readiness(summary, items, criteria)
    assert not result["gate_passed"]
    assert result["observed"]["success_by_domain_relation"]["place"]["P17"] == 1
    assert (
        result["observed"]["success_by_domain_relation"]["organization"]["P17"]
        == 0
    )


def test_freeze_rejects_failed_development_evidence(tmp_path):
    source, answers = _source(tmp_path)
    criteria = _criteria(minimum_qualified_per_domain=2)
    development = run_development_screening(
        _config(),
        source,
        "instrument_development",
        tmp_path / "development-output",
        runner=FakeRunner(answers),
        success_criteria=criteria,
        pre_model_semantic_audit=_semantic_audit(tmp_path, source),
        git_commit="a" * 40,
    )

    with pytest.raises(ValueError, match="development gate failed"):
        write_instrument_freeze_manifest(
            tmp_path / "freeze.json",
            source_root=source,
            development_run_dir=Path(development["summary_path"]).parent,
            config=_config(),
            success_criteria=criteria,
            git_commit="a" * 40,
        )


def test_freeze_rejects_success_criteria_for_another_source_revision(tmp_path):
    source, answers = _source(tmp_path)
    registered = _criteria()
    development = run_development_screening(
        _config(),
        source,
        "instrument_development",
        tmp_path / "development-output",
        runner=FakeRunner(answers),
        success_criteria=registered,
        pre_model_semantic_audit=_semantic_audit(tmp_path, source),
        git_commit="a" * 40,
    )
    criteria = {
        "schema_version": 2,
        "source_revision": "fa-development-source-v6-r3",
        "development_gate": {
            "candidate_count": 4,
            "prompt_count": 12,
            "qualification_threshold": 2,
            "minimum_qualified_per_domain": 1,
            "minimum_success_by_domain_relation": {
                domain: {
                    field.property_id: 1
                    for field in DEVELOPMENT_DOMAIN_FIELDS[domain]
                }
                for domain in REGISTERED_DOMAINS
            },
        },
    }

    with pytest.raises(ValueError, match="criteria source revision mismatch"):
        write_instrument_freeze_manifest(
            tmp_path / "freeze.json",
            source_root=source,
            development_run_dir=Path(development["summary_path"]).parent,
            config=_config(),
            success_criteria=criteria,
            git_commit="a" * 40,
        )


def test_freeze_reconstructs_derived_fields_from_raw_completion(tmp_path):
    source, answers = _source(tmp_path)
    criteria = _criteria()
    development = run_development_screening(
        _config(),
        source,
        "instrument_development",
        tmp_path / "development-output",
        runner=FakeRunner(answers),
        success_criteria=criteria,
        pre_model_semantic_audit=_semantic_audit(tmp_path, source),
        git_commit="a" * 40,
    )
    run_dir = Path(development["summary_path"]).parent
    items_path = run_dir / "screening_items.jsonl"
    rows = [
        json.loads(line)
        for line in items_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["is_correct"] = False
    items_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="batch shards"):
        write_instrument_freeze_manifest(
            tmp_path / "freeze.json",
            source_root=source,
            development_run_dir=run_dir,
            config=_config(),
            success_criteria=criteria,
            git_commit="a" * 40,
        )


def test_r4_freeze_requires_completed_manual_audit(tmp_path):
    source, answers = _source(tmp_path)
    criteria = _criteria(require_human_audit=True)
    development = run_development_screening(
        _config(),
        source,
        "instrument_development",
        tmp_path / "development-output",
        runner=FakeRunner(answers),
        success_criteria=criteria,
        pre_model_semantic_audit=_semantic_audit(tmp_path, source),
        git_commit="a" * 40,
    )

    with pytest.raises(ValueError, match="completed independent human audit"):
        write_instrument_freeze_manifest(
            tmp_path / "freeze.json",
            source_root=source,
            development_run_dir=Path(development["summary_path"]).parent,
            config=_config(),
            success_criteria=criteria,
            git_commit="a" * 40,
        )


def test_construction_validation_rejects_freeze_with_removed_manual_audit(
    tmp_path,
):
    source, answers = _source(tmp_path, candidates_per_domain_per_split=2)
    for domain in REGISTERED_DOMAINS:
        prompt = next(
            prompt
            for prompt in answers
            if domain.replace("_", " ").title().replace(" ", "_") in prompt
        )
        answers[prompt] = "wrong"
    criteria = _criteria(candidate_count=8, require_human_audit=True)
    development = run_development_screening(
        _config(),
        source,
        "instrument_development",
        tmp_path / "development-output",
        runner=FakeRunner(answers),
        success_criteria=criteria,
        pre_model_semantic_audit=_semantic_audit(tmp_path, source),
        git_commit="a" * 40,
    )
    manual_audit = _manual_audit(
        tmp_path / "manual-audit.json",
        development,
    )
    freeze = tmp_path / "freeze.json"
    write_instrument_freeze_manifest(
        freeze,
        source_root=source,
        development_run_dir=Path(development["summary_path"]).parent,
        config=_config(),
        success_criteria=criteria,
        manual_audit_manifest=manual_audit,
        git_commit="a" * 40,
    )
    payload = json.loads(freeze.read_text(encoding="utf-8"))
    del payload["manual_audit"]
    freeze.write_text(json.dumps(payload), encoding="utf-8")
    runner = FakeRunner(answers)

    with pytest.raises(ValueError, match="manual audit"):
        run_development_screening(
            _config(),
            source,
            "construction_validation",
            tmp_path / "validation-output",
            runner=runner,
            freeze_manifest=freeze,
            git_commit="a" * 40,
        )

    assert runner.calls == 0


def test_r5_screening_requires_passing_pre_model_semantic_audit(tmp_path):
    source, answers = _source(tmp_path)
    runner = FakeRunner(answers)

    with pytest.raises(ValueError, match="pre-model semantic audit"):
        run_development_screening(
            _config(),
            source,
            "instrument_development",
            tmp_path / "output",
            runner=runner,
            success_criteria=_criteria(),
            git_commit="a" * 40,
        )

    assert runner.calls == 0


def test_freeze_rejects_criteria_changed_after_development_run(tmp_path):
    source, answers = _source(tmp_path)
    registered = _criteria()
    development = run_development_screening(
        _config(),
        source,
        "instrument_development",
        tmp_path / "development-output",
        runner=FakeRunner(answers),
        success_criteria=registered,
        pre_model_semantic_audit=_semantic_audit(tmp_path, source),
        git_commit="a" * 40,
    )
    changed = _criteria(minimum_qualified_per_domain=2)

    with pytest.raises(ValueError, match="criteria hash"):
        write_instrument_freeze_manifest(
            tmp_path / "freeze.json",
            source_root=source,
            development_run_dir=Path(development["summary_path"]).parent,
            config=_config(),
            success_criteria=changed,
            git_commit="a" * 40,
        )


def test_freeze_rejects_completion_rewritten_outside_batch_shards(tmp_path):
    source, answers = _source(tmp_path)
    criteria = _criteria()
    development = run_development_screening(
        _config(),
        source,
        "instrument_development",
        tmp_path / "development-output",
        runner=FakeRunner(answers),
        success_criteria=criteria,
        pre_model_semantic_audit=_semantic_audit(tmp_path, source),
        git_commit="a" * 40,
    )
    run_dir = Path(development["summary_path"]).parent
    items_path = run_dir / "screening_items.jsonl"
    rows = [
        json.loads(line)
        for line in items_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0] = development_screening._item_row(
        development_screening._load_verified_source(
            source, "instrument_development"
        )["candidates"][0],
        development_screening._load_verified_source(
            source, "instrument_development"
        )["questions"][0],
        "fabricated completion",
        "instrument_development",
    )
    items_path.write_bytes(development_screening._jsonl_bytes(rows))

    with pytest.raises(ValueError, match="batch shards"):
        write_instrument_freeze_manifest(
            tmp_path / "freeze.json",
            source_root=source,
            development_run_dir=run_dir,
            config=_config(),
            success_criteria=criteria,
            git_commit="a" * 40,
        )


def test_construction_validation_reports_frozen_gate_result(tmp_path):
    source, answers = _source(tmp_path)
    freeze = _freeze(tmp_path, source, answers)

    result = run_development_screening(
        _config(),
        source,
        "construction_validation",
        tmp_path / "validation-output",
        runner=FakeRunner(answers),
        freeze_manifest=freeze,
        git_commit="a" * 40,
    )

    assert result["gate_result"]["gate_passed"] is True
    assert Path(result["gate_path"]).is_file()


def test_manual_audit_uses_registered_sampling_and_acceptance_rule(tmp_path):
    source, answers = _source(tmp_path, candidates_per_domain_per_split=2)
    for domain in REGISTERED_DOMAINS:
        prompt = next(
            prompt
            for prompt in answers
            if domain.replace("_", " ").title().replace(" ", "_") in prompt
        )
        answers[prompt] = "wrong"
    criteria = _criteria(candidate_count=8, require_human_audit=True)
    development = run_development_screening(
        _config(),
        source,
        "instrument_development",
        tmp_path / "development-output",
        runner=FakeRunner(answers),
        success_criteria=criteria,
        pre_model_semantic_audit=_semantic_audit(tmp_path, source),
        git_commit="a" * 40,
    )

    wrong_seed = _manual_audit(
        tmp_path / "wrong-seed.json",
        development,
        seed=1,
    )
    with pytest.raises(ValueError, match="registered sampling design"):
        write_instrument_freeze_manifest(
            tmp_path / "wrong-seed-freeze.json",
            source_root=source,
            development_run_dir=Path(development["summary_path"]).parent,
            config=_config(),
            success_criteria=criteria,
            manual_audit_manifest=wrong_seed,
            git_commit="a" * 40,
        )

    source_error = _manual_audit(
        tmp_path / "source-error.json",
        development,
        error_label="source_error",
    )
    with pytest.raises(ValueError, match="acceptance gate failed"):
        write_instrument_freeze_manifest(
            tmp_path / "source-error-freeze.json",
            source_root=source,
            development_run_dir=Path(development["summary_path"]).parent,
            config=_config(),
            success_criteria=criteria,
            manual_audit_manifest=source_error,
            git_commit="a" * 40,
        )

    scoring_mismatch = _manual_audit(
        tmp_path / "scoring-mismatch.json",
        development,
    )
    mismatch_payload = json.loads(scoring_mismatch.read_text(encoding="utf-8"))
    items = {
        row["question_id"]: row
        for row in (
            json.loads(line)
            for line in (
                Path(development["summary_path"]).parent / "screening_items.jsonl"
            )
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }
    success_row = next(
        row
        for row in mismatch_payload["packet"]
        if items[row["question_id"]]["is_correct"]
    )
    for rating in mismatch_payload["ratings"]:
        if rating["audit_id"] == success_row["audit_id"]:
            rating["error_label"] = "relation_unknown"
    mismatch_payload["compiled"] = compile_manual_error_audit(
        mismatch_payload["packet"],
        mismatch_payload["ratings"],
    )
    scoring_mismatch.write_text(
        json.dumps(mismatch_payload, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="scoring agreement gate"):
        write_instrument_freeze_manifest(
            tmp_path / "scoring-mismatch-freeze.json",
            source_root=source,
            development_run_dir=Path(development["summary_path"]).parent,
            config=_config(),
            success_criteria=criteria,
            manual_audit_manifest=scoring_mismatch,
            git_commit="a" * 40,
        )
