from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, replace
from functools import lru_cache

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

import trajectory_extractor.fa_probes as probes
from trajectory_extractor.fa_artifacts import FAArtifactStore, UnlockReceipt
from trajectory_extractor.fa_probes import (
    C_OPTIONS,
    DEFAULT_CONFIRMATORY_BOOTSTRAP_DRAWS,
    DEFAULT_FULL_SELECTION_NULL_SEED_HASH,
    DEFAULT_FULL_SELECTION_NULL_SEEDS,
    NESTED_H5_BASELINE,
    OUTPUT_CONTROL_SCHEMA_SHA256,
    PCA_OPTIONS,
    REGISTERED_LAYERS,
    BinaryMetrics,
    CrossConditionTransferSummary,
    F2AGates,
    GateCriterion,
    HypothesisGate,
    NullSelectionResult,
    ProbeRow,
    ProbeTestAuthorization,
    SAEGate,
    SelectionManifest,
    TARGET_FAMILIARITY_CONDITIONS,
    audit_sae_transfer,
    compute_binary_metrics,
    evaluate_f2a_gates,
    evaluate_probe_bundle_once,
    evaluate_probe_test_once,
    fit_selection,
    run_full_selection_nulls,
)


ANSWERABILITY_CLASSES = ("target_bound", "distractor_bound", "code_absent")
NESTED_H5_CANDIDATE = "surface_output_static"
NESTED_H6_CANDIDATE = "surface_output_static_dynamics"
REGISTERED_DOMAINS = ("person", "place", "organization", "creative_work")


@pytest.fixture(autouse=True)
def _fast_train_only_cv(monkeypatch):
    monkeypatch.setattr(probes, "_TRAIN_ONLY_CV_FAST_PATH_FOR_TESTS", True)
    monkeypatch.setattr(probes, "_BOOTSTRAP_DRAW_OVERRIDE_FOR_TESTS", 80)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


@lru_cache
def _rows(split: str, count: int = 12, *, task: str = "familiarity") -> tuple[ProbeRow, ...]:
    rows = []
    prefix = {"mechanism_train": "tr", "locked_validation": "va", "probe_test": "te"}[split]
    for index in range(count):
        answerability_condition = ANSWERABILITY_CLASSES[(index // 4) % 3]
        target_familiarity_condition = TARGET_FAMILIARITY_CONDITIONS[index % 2]
        distractor_familiarity_condition = TARGET_FAMILIARITY_CONDITIONS[(index // 2) % 2]
        if task == "answerability":
            label = answerability_condition
        elif task == "familiarity":
            label = int(target_familiarity_condition == "screened_real")
        else:
            label = index % 2
        signal = (
            float(ANSWERABILITY_CLASSES.index(label) - 1) * 2.0
            if task == "answerability"
            else (-2.0 if label == 0 else 2.0)
        )
        residual = np.zeros((3, 26, 4), dtype=np.float64)
        residual[:, :, 0] = signal
        residual[:, :, 1] = np.arange(26, dtype=np.float64)
        rows.append(
            ProbeRow(
                example_id=f"{prefix}-example-{index}",
                split=split,
                task=task,
                label=label,
                entity_id=f"{prefix}-entity-{(index // 2) % 2}",
                template_id=f"{prefix}-template-0",
                relation_id=f"{prefix}-relation-{index // 4}",
                domain=REGISTERED_DOMAINS[(index // 2) % 4],
                condition=f"condition-{(index // 2) % 4}",
                answerability_condition=answerability_condition,
                target_familiarity_condition=target_familiarity_condition,
                distractor_familiarity_condition=distractor_familiarity_condition,
                surface_features=(signal, float(index % 3)),
                output_margin_features=tuple([signal] + [float(index % 3)] * 10),
                residual_features=residual,
                sae_features=None,
                outcome_status="valid",
                source_sha256=_sha(f"source-{prefix}-{index}"),
                activation_sha256=_sha(f"activation-{prefix}-{index}"),
                metadata_manifest_sha256=_sha(f"metadata-manifest-{prefix}-{index}"),
                metadata_row_sha256=_sha(f"metadata-row-{prefix}-{index}"),
                output_control_schema_sha256=OUTPUT_CONTROL_SCHEMA_SHA256,
                output_evidence_sha256=_sha(f"output-evidence-{prefix}-{index}"),
            )
        )
    return tuple(rows)


class TrackingLogisticRegression(LogisticRegression):
    def __init__(self):
        super().__init__(solver="liblinear", random_state=17, max_iter=200)
        self.fit_ids: set[str] = set()


@lru_cache
def _selection(*, task: str = "familiarity") -> SelectionManifest:
    return fit_selection(
        _rows("mechanism_train", task=task),
        _rows("locked_validation", task=task),
    )


def _source_identities(rows: tuple[ProbeRow, ...]):
    return tuple(probes.ProbeSourceIdentity.from_row(row) for row in rows)


def _prompt_capability_record(rows: tuple[ProbeRow, ...]) -> dict[str, object]:
    return {
        "kind": "prompt_manifest",
        "config_hash": _sha("probe-config"),
        "full_manifest_sha256": _sha("full-prompt-manifest"),
        "subset_manifest_sha256": _sha("probe-subset-manifest"),
        "chat_template_sha256": _sha("chat-template"),
        "namespace": "probe_test",
        "model_sha256": _sha("model"),
        "tokenizer_sha256": _sha("tokenizer"),
        "tokenizer_pin_manifest": "runs/pins/tokenizer.jsonl.manifest.json",
        "tokenizer_pin_sha256": _sha("tokenizer-pin"),
        "naturalness_audit_manifest": "runs/audits/naturalness.jsonl.manifest.json",
        "naturalness_audit_sha256": _sha("naturalness-audit"),
        "generation": {"max_new_tokens": 16, "do_sample": False},
        "examples": [
            {
                "example_id": row.example_id,
                "canonical_payload_sha256": row.source_sha256,
                "split": "probe_test",
                "condition": row.condition,
            }
            for row in sorted(rows, key=lambda item: item.example_id)
        ],
    }


def _evaluate(
    root,
    selection: SelectionManifest,
    rows: tuple[ProbeRow, ...],
    *,
    null_selections=(),
):
    del root
    receipt = UnlockReceipt(
        endpoint="probe_test",
        lease_id="a" * 32,
        state="unlocked_once",
        preregistration_hash="b" * 64,
        selection_manifest_hash=selection.sha256,
    )
    authorization = ProbeTestAuthorization.from_unlock_receipt(receipt)
    identities = probes._canonical_source_identities(
        _source_identities(rows), field_name="test helper source identities"
    )
    return probes._calculate_probe_result(
        selection,
        authorization,
        rows,
        endpoint_input_sha256="c" * 64,
        endpoint_source_identities_sha256=probes._source_identity_digest(identities),
        expected_source_identities=identities,
        null_selections=null_selections,
    )


def _seal_bundle_endpoint(
    root,
    selections: Mapping[str, SelectionManifest],
    rows_by_task,
    *,
    record_kind: str = "prompt_manifest",
):
    store = FAArtifactStore(root)
    bundle_hash = probes.f2a_selection_bundle_hash(selections)
    source_rows = tuple(rows_by_task["familiarity"])
    expected = probes._canonical_source_identities(
        _source_identities(source_rows), field_name="test prompt source identities"
    )
    assert all(
        probes._canonical_source_identities(
            _source_identities(tuple(rows_by_task[task])), field_name="test task source identities"
        )
        == expected
        for task in probes.TASKS
    )
    shard = store.write_completed_shard(
        "probe-run",
        "probe_test",
        "probe-inputs",
        (_prompt_capability_record(source_rows),),
        {"selection_manifest": bundle_hash},
        record_kind=record_kind,
    )
    store.seal_endpoint(
        "probe_test",
        (shard,),
        {"preregistration": "b" * 64, "selection_manifest": bundle_hash},
    )
    return store, shard.manifest_path


def _bundle_authorization(root, selections: Mapping[str, SelectionManifest], rows_by_task):
    store, manifest_path = _seal_bundle_endpoint(root, selections, rows_by_task)
    bundle_hash = probes.f2a_selection_bundle_hash(selections)
    receipt = store.unlock_endpoint("probe_test", "b" * 64, bundle_hash)
    return store, manifest_path, ProbeTestAuthorization.from_unlock_receipt(receipt)


def _null_result(
    selection: SelectionManifest,
    kind: str,
    seed: int,
    *,
    selected_auroc: float | None = None,
    h6_improvement: float | None = None,
) -> NullSelectionResult:
    models = list(selection.models)
    if selected_auroc is not None:
        index = next(
            index
            for index, model in enumerate(models)
            if model.feature_family == selection.selected_feature_family
            and model.claim_scope == "pre_output"
        )
        models[index] = replace(models[index], validation_auroc=selected_auroc)
    if h6_improvement is not None:
        baseline_index = next(
            index
            for index, model in enumerate(models)
            if model.feature_family == NESTED_H5_CANDIDATE
            and model.claim_scope == "pre_output"
        )
        candidate_index = next(
            index
            for index, model in enumerate(models)
            if model.feature_family == NESTED_H6_CANDIDATE
            and model.claim_scope == "pre_output"
        )
        models[baseline_index] = replace(models[baseline_index], validation_log_loss=0.5)
        models[candidate_index] = replace(
            models[candidate_index], validation_log_loss=0.5 * (1.0 - h6_improvement)
        )
    provenance = {"kind": kind, "seed": seed, "config": {"test": True}}
    null_selection = replace(
        selection,
        models=tuple(models),
        null_provenance=provenance,
    )
    config = {"kind": kind, "seed": seed, "transform": {"test": True}}
    config_sha256 = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return NullSelectionResult(
        kind=kind,
        seed=seed,
        config=config,
        config_sha256=config_sha256,
        selection=null_selection,
        max_norm_error=0.0,
        test_source_identities=_source_identities(
            _rows("probe_test", task=selection.task)
        ),
        test_transform={"seed": seed, "row_count": len(_rows("probe_test"))},
    )


def _score_null(
    null: NullSelectionResult,
    result: probes.ProbeResult,
    *,
    auroc: float | None = None,
    h6_improvement: float | None = None,
) -> NullSelectionResult:
    metrics = result.metrics if auroc is None else replace(result.metrics, auroc=auroc)
    transfer = result.cross_condition_transfer
    if transfer is not None and auroc is not None:
        transfer = replace(
            transfer,
            rotations=tuple(
                replace(rotation, metrics=replace(rotation.metrics, auroc=auroc))
                for rotation in transfer.rotations
            ),
        )
    return replace(
        null,
        test_ids=result.test_ids,
        test_row_sha256s=result.test_row_sha256s,
        test_metrics=metrics,
        test_model_metrics={},
        test_cross_condition_transfer=transfer,
        test_relative_h5_log_loss_improvement=result.relative_h5_log_loss_improvement,
        test_relative_h6_log_loss_improvement=h6_improvement,
    )


def test_registered_grid_is_exact_and_records_are_deeply_immutable():
    assert REGISTERED_LAYERS == tuple(range(26))
    assert PCA_OPTIONS == (None, 16, 32, 64)
    assert C_OPTIONS == (0.01, 0.1, 1.0, 10.0)
    row = _rows("mechanism_train", 2)[0]
    with pytest.raises(FrozenInstanceError):
        row.label = 0
    with pytest.raises(ValueError):
        row.residual_features[0, 0, 0] = 99.0
    with pytest.raises(ValueError):
        row.residual_features.setflags(write=True)
    mutable = np.zeros((3, 26, 4), dtype=np.float64)
    copied = replace(row, residual_features=mutable)
    before = copied.sha256
    mutable[0, 0, 0] = 99.0
    assert copied.sha256 == before
    record = copied.to_record()
    assert record["metadata_manifest_sha256"] == copied.metadata_manifest_sha256
    assert record["metadata_row_sha256"] == copied.metadata_row_sha256
    gate = SAEGate(1.0, 1.2, 2.0, 0.95, 0.8, [])
    assert gate.reasons == ()
    with pytest.raises(ValueError, match="recovery"):
        SAEGate(1.0, 1.2, 2.0, 0.95, 0.1, ())


def test_transform_and_estimator_fit_only_on_mechanism_train():
    tracked = TrackingLogisticRegression()
    train = _rows("mechanism_train")
    validation = _rows("locked_validation")
    selection = fit_selection(
        train,
        validation,
        estimators=(tracked,),
    )
    assert tracked.fit_ids == {row.example_id for row in train}
    assert not tracked.fit_ids & {row.example_id for row in validation}
    assert selection.train_ids == tuple(sorted(row.example_id for row in train))


def test_selection_persists_exact_registered_transfer_rotations():
    familiarity = _selection(task="familiarity")
    assert len(familiarity.transfer_rotations) == 3
    assert {
        (rotation.train_condition, rotation.test_conditions)
        for rotation in familiarity.transfer_rotations
    } == {
        (condition, tuple(other for other in ANSWERABILITY_CLASSES if other != condition))
        for condition in ANSWERABILITY_CLASSES
    }
    assert all(
        rotation.model.feature_family == familiarity.selected_feature_family
        for rotation in familiarity.transfer_rotations
    )
    assert all(
        isinstance(score.cross_condition_transfer, CrossConditionTransferSummary)
        for score in familiarity.candidate_scores
        if score.status == "evaluable"
    )

    answerability = _selection(task="answerability")
    assert len(answerability.transfer_rotations) == 2
    assert {
        (rotation.train_condition, rotation.test_conditions)
        for rotation in answerability.transfer_rotations
    } == {
        ("screened_real", ("matched_synthetic",)),
        ("matched_synthetic", ("screened_real",)),
    }
    record = answerability.to_record()
    assert len(record["cross_condition_transfer_rotations"]) == 2
    assert all(
        score["cross_condition_transfer"] is not None
        for score in record["candidate_scores"]
        if score["status"] == "evaluable"
    )


def test_selection_rejects_missing_duplicate_and_unregistered_transfer_rotations():
    selection = _selection(task="familiarity")
    rotations = selection.transfer_rotations
    with pytest.raises(ValueError, match="exact registered transfer rotations"):
        replace(selection, transfer_rotations=rotations[:-1])
    with pytest.raises(ValueError, match="exact registered transfer rotations"):
        replace(selection, transfer_rotations=(rotations[0], rotations[0], rotations[2]))
    with pytest.raises(ValueError, match="registered train condition"):
        replace(rotations[0], train_condition="unregistered")


def test_fit_selection_rejects_protected_splits_ids_and_group_leakage():
    train = _rows("mechanism_train")
    validation = _rows("locked_validation")
    with pytest.raises(ValueError, match="protected split"):
        fit_selection(_rows("probe_test"), validation)
    with pytest.raises(ValueError, match="protected test ID"):
        fit_selection(train, validation, protected_test_ids={train[0].example_id})
    leaked = replace(validation[0], entity_id=train[0].entity_id)
    with pytest.raises(ValueError, match="entity leakage"):
        fit_selection(train, (leaked, *validation[1:]))
    leaked_template = replace(validation[0], template_id=train[0].template_id)
    with pytest.raises(ValueError, match="template leakage"):
        fit_selection(train, (leaked_template, *validation[1:]))


def test_registered_domains_overlap_across_splits_and_are_reporting_strata_only(tmp_path):
    train = _rows("mechanism_train")
    validation = _rows("locked_validation")
    test = _rows("probe_test")
    assert {row.domain for row in train} == set(REGISTERED_DOMAINS)
    assert {row.domain for row in validation} == set(REGISTERED_DOMAINS)
    assert {row.domain for row in test} == set(REGISTERED_DOMAINS)
    selection = fit_selection(train, validation)
    result = _evaluate(tmp_path, selection, test)
    assert set(selection.train_domain_ids) == set(REGISTERED_DOMAINS)
    assert set(selection.validation_domain_ids) == set(REGISTERED_DOMAINS)
    assert set(result.ood_transfer["domain"]) == set(REGISTERED_DOMAINS)


def test_output_control_is_schema_and_evidence_provenance_bound():
    row = _rows("mechanism_train")[0]
    assert row.output_control_schema_sha256 == OUTPUT_CONTROL_SCHEMA_SHA256
    assert row.to_record()["output_evidence_sha256"] == row.output_evidence_sha256
    with pytest.raises(ValueError, match="output control schema"):
        replace(row, output_control_schema_sha256=_sha("wrong-output-schema"))
    with pytest.raises(ValueError, match="output_evidence_sha256"):
        replace(row, output_evidence_sha256="not-a-sha")


def test_output_margin_is_exactly_eleven_dimensional():
    bad = replace(_rows("mechanism_train")[0], output_margin_features=(0.0,) * 10)
    with pytest.raises(ValueError, match="exactly 11"):
        fit_selection((bad, *_rows("mechanism_train")[1:]), _rows("locked_validation"))


def test_callers_cannot_narrow_the_registered_candidate_grid_or_baselines():
    train = _rows("mechanism_train")
    validation = _rows("locked_validation")
    with pytest.raises(ValueError, match="exact registered PCA grid"):
        fit_selection(train, validation, pca_options=(None,))
    with pytest.raises(ValueError, match="exact registered C grid"):
        fit_selection(train, validation, c_options=(1.0,))
    with pytest.raises(ValueError, match="required baseline"):
        fit_selection(train, validation, feature_families=("surface",))
    selection = fit_selection(train, validation)
    assert selection.pca_options == PCA_OPTIONS
    assert selection.c_options == C_OPTIONS
    residual_scores = [
        score for score in selection.candidate_scores if score.feature_family == "residual_static"
    ]
    assert {score.layer for score in residual_scores} == set(range(26))
    assert {score.pca_components for score in residual_scores} == set(PCA_OPTIONS)
    assert {score.c for score in residual_scores} == set(C_OPTIONS)


def test_task_anchor_is_registered_and_output_proximal_is_control_only():
    familiarity = _selection(task="familiarity")
    assert {model.anchor for model in familiarity.models} == {"target_intro_end"}
    unsupported = _selection(task="unsupported_answer")
    assert {model.anchor for model in unsupported.models} == {
        "user_prompt_end",
        "assistant_prefix_end",
    }
    assert all(
        model.claim_scope == "output_proximal_control"
        for model in unsupported.models
        if model.anchor == "assistant_prefix_end"
    )


def test_probe_result_persists_selection_bound_model_scope_evidence(tmp_path):
    selection = _selection(task="familiarity")
    result = _evaluate(tmp_path, selection, _rows("probe_test", task="familiarity"))
    selected = selection.model_for(selection.selected_feature_family)
    assert result.selected_feature_family == selected.feature_family
    assert result.selected_anchor == selected.anchor
    assert result.selected_layer == selected.layer
    assert result.claim_scope == "pre_output"
    assert result.selected_model_sha256 == selected.sha256
    assert result.to_record()["selected_model_scope"] == {
        "feature_family": selected.feature_family,
        "anchor": selected.anchor,
        "layer": selected.layer,
        "claim_scope": "pre_output",
        "selected_model_sha256": selected.sha256,
    }


def test_final_layer_excluded_never_selects_layer_25():
    selection = _selection()
    model = selection.model_for("final_layer_excluded")
    assert model.layer in range(25)


def test_probe_authorization_requires_durable_store_state_and_rejects_forgery(tmp_path):
    selections = {task: _selection(task=task) for task in probes.TASKS}
    rows_by_task = {task: _rows("probe_test", task=task) for task in probes.TASKS}
    store, manifest_path, authorization = _bundle_authorization(tmp_path, selections, rows_by_task)
    forged_receipt = UnlockReceipt(
        endpoint="probe_test",
        lease_id="a" * 32,
        state="unlocked_once",
        preregistration_hash="b" * 64,
        selection_manifest_hash=probes.f2a_selection_bundle_hash(selections),
    )
    forged = ProbeTestAuthorization.from_unlock_receipt(forged_receipt)
    with pytest.raises(ValueError, match="endpoint lease"):
        evaluate_probe_bundle_once(
            selections,
            forged,
            rows_by_task,
            store=store,
            endpoint_manifest_path=manifest_path,
        )
    with pytest.raises(ValueError, match="FAArtifactStore"):
        evaluate_probe_bundle_once(selections, authorization, rows_by_task)


def test_single_task_probe_endpoint_fails_closed_before_lease_or_write(tmp_path):
    selection = _selection(task="familiarity")
    rows = _rows("probe_test", task="familiarity")
    store = FAArtifactStore(tmp_path)
    shard = store.write_completed_shard(
        "probe-run",
        "probe_test",
        "probe-inputs",
        (_prompt_capability_record(rows),),
        {"selection_manifest": selection.sha256},
        record_kind="prompt_manifest",
    )
    store.seal_endpoint(
        "probe_test",
        (shard,),
        {"preregistration": "b" * 64, "selection_manifest": selection.sha256},
    )
    authorization = ProbeTestAuthorization.from_unlock_receipt(
        UnlockReceipt(
            endpoint="probe_test",
            lease_id="a" * 32,
            state="unlocked_once",
            preregistration_hash="b" * 64,
            selection_manifest_hash=selection.sha256,
        )
    )

    assert store.endpoint_state("probe_test", shard.manifest_path) == "sealed"
    with pytest.raises(ValueError, match="bundle-only"):
        evaluate_probe_test_once(
            selection,
            authorization,
            rows,
            store=store,
            endpoint_manifest_path=shard.manifest_path,
        )
    assert store.endpoint_state("probe_test", shard.manifest_path) == "sealed"
    assert not tuple((tmp_path / "runs" / "familiarity_answerability").glob("**/*metrics*"))


def test_probe_endpoint_requires_verified_prompt_manifest_record_kind(tmp_path):
    selections = {task: _selection(task=task) for task in probes.TASKS}
    rows_by_task = {task: _rows("probe_test", task=task) for task in probes.TASKS}
    store, manifest_path = _seal_bundle_endpoint(
        tmp_path, selections, rows_by_task, record_kind="generic"
    )
    receipt = store.unlock_endpoint(
        "probe_test", "b" * 64, probes.f2a_selection_bundle_hash(selections)
    )
    authorization = ProbeTestAuthorization.from_unlock_receipt(receipt)
    with pytest.raises(ValueError, match="prompt_manifest"):
        evaluate_probe_bundle_once(
            selections,
            authorization,
            rows_by_task,
            store=store,
            endpoint_manifest_path=manifest_path,
        )


@pytest.mark.parametrize(
    ("mutate_records", "error"),
    (
        (lambda record: (record, record), "exactly one"),
        (
            lambda record: ({**record, "examples": {"not": "a list"}},),
            "examples list",
        ),
        (
            lambda record: (
                {**record, "examples": [*record["examples"], record["examples"][0]]},
            ),
            "duplicate source identities",
        ),
    ),
)
def test_probe_endpoint_prompt_manifest_requires_one_canonical_source_list(
    tmp_path, mutate_records, error
):
    selections = {task: _selection(task=task) for task in probes.TASKS}
    rows_by_task = {task: _rows("probe_test", task=task) for task in probes.TASKS}
    store = FAArtifactStore(tmp_path)
    bundle_hash = probes.f2a_selection_bundle_hash(selections)
    records = mutate_records(_prompt_capability_record(rows_by_task["familiarity"]))
    shard = store.write_completed_shard(
        "probe-run",
        "probe_test",
        "probe-inputs",
        records,
        {"selection_manifest": bundle_hash},
        record_kind="prompt_manifest",
    )
    store.seal_endpoint(
        "probe_test",
        (shard,),
        {"preregistration": "b" * 64, "selection_manifest": bundle_hash},
    )
    authorization = ProbeTestAuthorization.from_unlock_receipt(
        store.unlock_endpoint("probe_test", "b" * 64, bundle_hash)
    )
    with pytest.raises(ValueError, match=error):
        evaluate_probe_bundle_once(
            selections,
            authorization,
            rows_by_task,
            store=store,
            endpoint_manifest_path=shard.manifest_path,
        )


def test_selection_is_hash_bound_frozen_and_probe_test_consumption_is_durable(tmp_path):
    selections = {task: _selection(task=task) for task in probes.TASKS}
    rows_by_task = {task: _rows("probe_test", task=task) for task in probes.TASKS}
    store, manifest_path, authorization = _bundle_authorization(tmp_path, selections, rows_by_task)
    result = evaluate_probe_bundle_once(
        selections,
        authorization,
        rows_by_task,
        store=store,
        endpoint_manifest_path=manifest_path,
    )
    assert result.selection_bundle_hash == probes.f2a_selection_bundle_hash(selections)
    assert result.refit_performed is False
    assert store.endpoint_state("probe_test", manifest_path) == "closed"
    restarted_store = FAArtifactStore(tmp_path)
    with pytest.raises(ValueError, match="already closed"):
        evaluate_probe_bundle_once(
            selections,
            authorization,
            rows_by_task,
            store=restarted_store,
            endpoint_manifest_path=manifest_path,
        )
    changed = dict(selections)
    changed["familiarity"] = replace(selections["familiarity"], seed=17)
    with pytest.raises(ValueError, match="selection hash"):
        evaluate_probe_bundle_once(
            changed,
            authorization,
            rows_by_task,
            store=restarted_store,
            endpoint_manifest_path=manifest_path,
        )


@pytest.mark.parametrize(
    ("rows", "error"),
    (
        (lambda rows: (*rows[1:],), "sealed probe-test source identities"),
        (
            lambda rows: (
                replace(rows[0], source_sha256=_sha("substituted-source")),
                *rows[1:],
            ), "sealed probe-test source identities"),
        (
            lambda rows: (
                *rows,
                replace(
                    rows[0],
                    example_id="te-extra",
                    entity_id="te-extra-entity",
                    template_id="te-extra-template",
                    relation_id="te-extra-relation",
                    domain="te-extra-domain",
                ),
            ), "sealed probe-test source identities"),
        (lambda rows: (*rows, rows[0]), "duplicate"),
    ),
)
def test_probe_test_requires_exact_sealed_source_identity_multiset(tmp_path, rows, error):
    selections = {task: _selection(task=task) for task in probes.TASKS}
    rows_by_task = {task: _rows("probe_test", task=task) for task in probes.TASKS}
    store, manifest_path, authorization = _bundle_authorization(tmp_path, selections, rows_by_task)
    changed = dict(rows_by_task)
    changed["familiarity"] = rows(rows_by_task["familiarity"])
    with pytest.raises(ValueError, match=error):
        evaluate_probe_bundle_once(
            selections,
            authorization,
            changed,
            store=store,
            endpoint_manifest_path=manifest_path,
        )


def test_prompt_manifest_does_not_precommit_protected_probe_row_hashes(tmp_path):
    selections = {task: _selection(task=task) for task in probes.TASKS}
    rows_by_task = {task: _rows("probe_test", task=task) for task in probes.TASKS}
    store, manifest_path, authorization = _bundle_authorization(
        tmp_path, selections, rows_by_task
    )
    changed = dict(rows_by_task)
    changed["familiarity"] = (
        replace(
            rows_by_task["familiarity"][0],
            activation_sha256=_sha("protected-activation-built-after-open"),
            output_evidence_sha256=_sha("protected-output-built-after-open"),
        ),
        *rows_by_task["familiarity"][1:],
    )
    bundle = evaluate_probe_bundle_once(
        selections,
        authorization,
        changed,
        store=store,
        endpoint_manifest_path=manifest_path,
    )
    assert changed["familiarity"][0].sha256 in bundle.results[
        "familiarity"
    ].test_row_sha256s


def test_probe_metrics_lineage_binds_verified_endpoint_source_identities(tmp_path):
    selections = {task: _selection(task=task) for task in probes.TASKS}
    rows_by_task = {task: _rows("probe_test", task=task) for task in probes.TASKS}
    store, manifest_path, authorization = _bundle_authorization(tmp_path, selections, rows_by_task)
    result = evaluate_probe_bundle_once(
        selections,
        authorization,
        rows_by_task,
        store=store,
        endpoint_manifest_path=manifest_path,
    )
    endpoint = store.verify_endpoint_artifact("probe_test", manifest_path)
    metrics_manifest = next(
        endpoint.data_path.parent.glob(
            f"probe-bundle-metrics-{authorization.lease_id}.jsonl.manifest.json"
        )
    )
    metrics = json.loads(metrics_manifest.read_text(encoding="utf-8"))
    assert metrics["lineage"]["endpoint_input_sha256"] == endpoint.sha256
    source_identities = probes._canonical_source_identities(
        _source_identities(rows_by_task["familiarity"]),
        field_name="expected source identities",
    )
    expected_digest = _sha(
        json.dumps(
            [identity.to_record() for identity in source_identities],
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    assert metrics["lineage"]["endpoint_source_identities_sha256"] == expected_digest
    assert result.endpoint_input_sha256 == endpoint.sha256
    assert result.endpoint_source_identities_sha256 == expected_digest
    assert all(
        task_result.endpoint_source_identities_sha256 == expected_digest
        for task_result in result.results.values()
    )


def test_metrics_track_denominators_ties_and_fail_closed_on_single_class():
    tied = compute_binary_metrics(
        labels=(0, 1, 0, 1),
        probabilities=(0.5, 0.5, 0.5, 0.5),
        threshold=0.5,
        total=6,
        missing=1,
        invalid=1,
    )
    assert isinstance(tied, BinaryMetrics)
    assert tied.denominator == 4
    assert (tied.total, tied.missing, tied.invalid) == (6, 1, 1)
    assert tied.auroc == pytest.approx(0.5)
    single = compute_binary_metrics((1, 1), (0.8, 0.9), threshold=0.5)
    assert single.status == "not_evaluable"
    assert single.auroc is None
    assert any("single class" in reason for reason in single.reasons)


def test_answerability_is_three_state_and_uses_macro_multiclass_metrics(tmp_path):
    rows = _rows("mechanism_train", task="answerability")
    assert {row.label for row in rows} == set(ANSWERABILITY_CLASSES)
    with pytest.raises(ValueError, match="three-state"):
        replace(rows[0], label=1)
    metrics = probes.compute_classification_metrics(
        (
            "target_bound",
            "distractor_bound",
            "code_absent",
            "target_bound",
            "distractor_bound",
            "code_absent",
        ),
        (
            (0.8, 0.1, 0.1),
            (0.1, 0.8, 0.1),
            (0.1, 0.1, 0.8),
            (0.7, 0.2, 0.1),
            (0.2, 0.7, 0.1),
            (0.2, 0.1, 0.7),
        ),
        classes=ANSWERABILITY_CLASSES,
    )
    assert metrics.classes == ANSWERABILITY_CLASSES
    assert metrics.auroc == pytest.approx(1.0)
    assert metrics.balanced_accuracy == pytest.approx(1.0)
    assert metrics.log_loss is not None
    assert metrics.calibration_error is not None
    empty = probes.compute_classification_metrics(
        (),
        (),
        classes=ANSWERABILITY_CLASSES,
        total=1,
        missing=1,
    )
    assert empty.status == "not_evaluable"
    assert empty.denominator == 0
    selection = _selection(task="answerability")
    frozen = selection.model_for("surface")
    validation = _rows("locked_validation", task="answerability")
    matrix = np.asarray([row.surface_features for row in validation])
    scaled = (matrix - np.asarray(frozen.scaler_mean)) / np.asarray(frozen.scaler_scale)
    logits = scaled @ np.asarray(frozen.coefficients).T + np.asarray(frozen.intercepts)
    exponentials = np.exp(logits - logits.max(axis=1, keepdims=True))
    assert frozen.predict_proba(validation) == pytest.approx(
        exponentials / exponentials.sum(axis=1, keepdims=True)
    )
    result = _evaluate(tmp_path, selection, _rows("probe_test", task="answerability"))
    assert result.metrics.classes == ANSWERABILITY_CLASSES
    assert result.metrics.threshold is None


@pytest.mark.parametrize(("task", "expected_rotations"), (("familiarity", 3), ("answerability", 2)))
def test_confirmatory_result_serializes_each_reciprocal_test_and_distractor_cell(
    tmp_path, task, expected_rotations
):
    result = _evaluate(tmp_path, _selection(task=task), _rows("probe_test", task=task))
    transfer = result.cross_condition_transfer
    assert isinstance(transfer, CrossConditionTransferSummary)
    assert len(transfer.rotations) == expected_rotations
    for rotation in transfer.rotations:
        assert tuple(item.test_condition for item in rotation.condition_results) == rotation.test_conditions
        for condition in rotation.condition_results:
            assert tuple(
                cell.distractor_familiarity_condition
                for cell in condition.distractor_familiarity_cells
            ) == TARGET_FAMILIARITY_CONDITIONS
    expected_mean = sum(rotation.metrics.auroc for rotation in transfer.rotations) / expected_rotations
    expected_worst = min(
        cell.metrics.balanced_accuracy
        for rotation in transfer.rotations
        for condition in rotation.condition_results
        for cell in condition.distractor_familiarity_cells
    )
    assert transfer.aggregation == "equal_weight_mean_across_registered_rotations"
    assert transfer.mean_auroc == pytest.approx(expected_mean)
    assert transfer.worst_cell_balanced_accuracy == pytest.approx(expected_worst)
    record = result.to_record()["cross_condition_transfer"]
    assert len(record["rotations"]) == expected_rotations
    assert all(len(rotation["condition_results"]) == len(rotation["test_conditions"]) for rotation in record["rotations"])


def test_confirmatory_transfer_result_rejects_missing_duplicate_and_unregistered_rotations(tmp_path):
    result = _evaluate(
        tmp_path,
        _selection(task="familiarity"),
        _rows("probe_test", task="familiarity"),
    )
    transfer = result.cross_condition_transfer
    assert transfer is not None
    with pytest.raises(ValueError, match="exact registered transfer rotations"):
        replace(transfer, rotations=transfer.rotations[:-1])
    with pytest.raises(ValueError, match="exact registered transfer rotations"):
        replace(
            transfer,
            rotations=(transfer.rotations[0], transfer.rotations[0], transfer.rotations[2]),
        )
    with pytest.raises(ValueError, match="registered train condition"):
        replace(transfer.rotations[0], train_condition="unregistered")
    with pytest.raises(ValueError, match="requires reciprocal cross-condition transfer"):
        replace(result, cross_condition_transfer=None)


def test_test_result_includes_condition_domain_and_ood_worst_case_metrics(tmp_path):
    selection = _selection()
    result = _evaluate(tmp_path, selection, _rows("probe_test"))
    assert set(result.per_condition) == {f"condition-{index}" for index in range(4)}
    assert result.worst_condition is not None
    assert set(result.ood_transfer) == {"entity", "template", "relation", "domain"}
    assert set(result.worst_ood_transfer) == {"entity", "template", "relation", "domain"}
    assert result.ood_transfer["template"]
    assert result.ood_transfer["domain"]


def test_group_metrics_retain_local_missing_and_invalid_denominators(tmp_path):
    selection = _selection()
    rows = list(_rows("probe_test"))
    rows[0] = replace(rows[0], outcome_status="missing")
    rows[1] = replace(rows[1], outcome_status="invalid")
    result = _evaluate(tmp_path, selection, tuple(rows))
    assert (result.metrics.total, result.metrics.denominator) == (12, 10)
    assert (result.metrics.missing, result.metrics.invalid) == (1, 1)
    condition = result.per_condition["condition-0"]
    assert (condition.total, condition.denominator) == (4, 2)
    assert (condition.missing, condition.invalid) == (1, 1)


def test_single_class_condition_makes_worst_case_gate_fail_closed(tmp_path):
    selection = _selection()
    rows = list(_rows("probe_test"))
    rows[0] = replace(rows[0], label=0)
    rows[8] = replace(rows[8], label=0)
    result = _evaluate(tmp_path, selection, tuple(rows))
    assert result.per_condition["condition-0"].status == "not_evaluable"
    assert result.worst_condition is None
    assert result.primary_gate.status == "not_evaluable"


def test_sae_transfer_rejects_invalids_and_failure_is_nonblocking():
    passed = audit_sae_transfer(1.0, 1.2, 2.0, 0.95)
    assert passed.passed is True
    failed = audit_sae_transfer(1.0, 1.4, 2.0, 0.94)
    assert failed.passed is False
    assert failed.blocking is False
    with pytest.raises(ValueError, match="denominator"):
        audit_sae_transfer(1.0, 1.1, 1.0, 1.0)
    with pytest.raises(ValueError, match="finite"):
        audit_sae_transfer(1.0, np.nan, 2.0, 1.0)


def test_sae_candidates_require_a_passing_transfer_audit():
    train = tuple(replace(row, sae_features=np.copy(row.residual_features)) for row in _rows("mechanism_train"))
    validation = tuple(replace(row, sae_features=np.copy(row.residual_features)) for row in _rows("locked_validation"))
    failed = audit_sae_transfer(1.0, 1.4, 2.0, 0.94)
    without_sae = fit_selection(train, validation, sae_gate=failed)
    assert not ({model.feature_family for model in without_sae.models} & {"sae_1_sparse", "sae_small_sparse"})
    passed = audit_sae_transfer(1.0, 1.2, 2.0, 0.95)
    selection = fit_selection(
        train,
        validation,
        sae_gate=passed,
    )
    assert selection.model_for("sae_1_sparse").feature_family == "sae_1_sparse"


def test_sae_identity_is_selected_by_train_only_cv_and_small_subset_is_at_most_five():
    train = []
    for index, row in enumerate(_rows("mechanism_train")):
        features = np.zeros((3, 26, 8), dtype=np.float64)
        features[:, :, 0] = index * 100.0
        features[:, :, 6] = index % 2
        train.append(replace(row, sae_features=features))
    validation = []
    for index, row in enumerate(_rows("locked_validation")):
        features = np.zeros((3, 26, 8), dtype=np.float64)
        features[:, :, 0] = 1.0
        features[:, :, 6] = index * 100.0
        validation.append(replace(row, sae_features=features))
    selection = fit_selection(
        tuple(train),
        tuple(validation),
        sae_gate=audit_sae_transfer(1.0, 1.2, 2.0, 0.95),
    )
    assert selection.model_for("sae_1_sparse").selector_indices == (0,)
    assert len(selection.model_for("sae_small_sparse").selector_indices) <= 5


def test_h5_and_h6_use_exact_registered_nested_models(tmp_path):
    selection = _selection(task="unsupported_answer")
    h5_baseline = selection.model_for(NESTED_H5_BASELINE)
    h5_candidate = selection.model_for(NESTED_H5_CANDIDATE)
    h6_candidate = selection.model_for(NESTED_H6_CANDIDATE)
    assert h5_baseline.claim_scope == "pre_output"
    assert h5_candidate.claim_scope == "pre_output"
    assert h6_candidate.claim_scope == "pre_output"
    assert h5_baseline.anchor != "assistant_prefix_end"
    assert h5_candidate.anchor != "assistant_prefix_end"
    assert h6_candidate.anchor != "assistant_prefix_end"
    assert h5_candidate.layer != 25
    assert h6_candidate.layer != 25
    assert len(h5_baseline.scaler_mean) == 2
    assert len(h5_candidate.scaler_mean) == 2 + 4
    assert len(h6_candidate.scaler_mean) == 2 + 4 + 4 + 1
    result = _evaluate(tmp_path, selection, _rows("probe_test", task="unsupported_answer"))
    expected_h5 = (
        result.model_metrics[NESTED_H5_BASELINE].log_loss
        - result.model_metrics[NESTED_H5_CANDIDATE].log_loss
    ) / result.model_metrics[NESTED_H5_BASELINE].log_loss
    expected_h6 = (
        result.model_metrics[NESTED_H5_CANDIDATE].log_loss
        - result.model_metrics[NESTED_H6_CANDIDATE].log_loss
    ) / result.model_metrics[NESTED_H5_CANDIDATE].log_loss
    assert result.relative_h5_log_loss_improvement == pytest.approx(expected_h5)
    assert result.relative_h6_log_loss_improvement == pytest.approx(expected_h6)
    assert result.h5_absolute_log_loss_difference_95 is not None
    assert result.h6_absolute_log_loss_difference_95 is not None


def test_all_nulls_rerun_full_selection_and_record_seed_config(monkeypatch):
    import trajectory_extractor.fa_probes as probes

    calls = []
    base_selection = _selection()

    def tracked_fit(*args, **kwargs):
        calls.append(kwargs.get("null_provenance"))
        return replace(base_selection, null_provenance=kwargs.get("null_provenance"))

    monkeypatch.setattr(probes, "fit_selection", tracked_fit)
    nulls = run_full_selection_nulls(
        _rows("mechanism_train"),
        _rows("locked_validation"),
        seeds=(7,),
        _allow_test_seed_override=True,
        probe_test_source_identities=_source_identities(_rows("probe_test")),
    )
    assert len(nulls) == 4
    assert len(calls) == 4
    assert {item.kind for item in nulls} == {
        "label_permutation",
        "layer_order",
        "random_map",
        "output_aligned_11d",
    }
    assert all(item.seed == 7 and item.config_sha256 for item in nulls)
    assert all(item.selection.null_provenance is not None for item in nulls)
    assert all(item.test_source_identities for item in nulls)
    assert all(
        set(item.test_transform) == {"seed", "row_count"}
        and item.test_transform["seed"] == 7
        and item.test_transform["row_count"] == len(_rows("probe_test"))
        for item in nulls
    )
    assert all(
        set(identity.to_record()) == {"example_id", "canonical_payload_sha256"}
        for item in nulls
        for identity in item.test_source_identities
    )
    with pytest.raises(ValueError, match="row_count"):
        replace(nulls[0], test_transform={"seed": 7, "row_count": 11})


def test_confirmatory_full_selection_null_seeds_are_fixed_and_hash_bound(monkeypatch):
    assert DEFAULT_FULL_SELECTION_NULL_SEEDS == tuple(range(2026072201, 2026072300))
    assert DEFAULT_FULL_SELECTION_NULL_SEED_HASH == "7aee4f4ee03201f4a8b7bee296294bc5c6a14a5251dfa71bb8cff15ce3d4e07f"
    with pytest.raises(ValueError, match="registered full-selection null seeds"):
        run_full_selection_nulls(
            _rows("mechanism_train"),
            _rows("locked_validation"),
            seeds=(2026072201,),
            probe_test_source_identities=_source_identities(_rows("probe_test")),
        )


def test_frozen_full_selection_nulls_are_scored_on_the_same_sealed_probe_test_rows(tmp_path):
    selection = _selection()
    rows = _rows("probe_test")
    nulls = run_full_selection_nulls(
        _rows("mechanism_train"),
        _rows("locked_validation"),
        seeds=(7,),
        _allow_test_seed_override=True,
        probe_test_source_identities=_source_identities(rows),
    )
    result = _evaluate(tmp_path, selection, rows, null_selections=nulls)
    assert {null.kind for null in result.null_results} == {
        "label_permutation",
        "layer_order",
        "random_map",
        "output_aligned_11d",
    }
    assert all(null.test_metrics is not None for null in result.null_results)
    assert all(null.test_ids == result.test_ids for null in result.null_results)
    assert all(null.test_row_sha256s for null in result.null_results)
    assert all(
        null.test_source_identities
        == probes._canonical_source_identities(
            _source_identities(rows), field_name="expected null source identities"
        )
        for null in result.null_results
    )
    label_permutation = next(null for null in result.null_results if null.kind == "label_permutation")
    assert label_permutation.test_row_sha256s != result.test_row_sha256s


def test_bootstrap_records_requested_valid_discarded_counts_and_seed(monkeypatch, tmp_path):
    monkeypatch.setattr(probes, "_BOOTSTRAP_DRAW_OVERRIDE_FOR_TESTS", 80)
    selection = _selection()
    result = _evaluate(tmp_path, selection, _rows("probe_test"))
    interval = result.crossed_auroc_95
    assert interval is not None
    assert interval.requested_draws == DEFAULT_CONFIRMATORY_BOOTSTRAP_DRAWS
    assert interval.valid_draws + interval.discarded_draws == 80
    assert interval.valid_draws == interval.draws
    assert interval.seed == 20260722
    assert interval.resampling_unit == "crossed_entity_template"


def test_dynamics_use_adjacent_difference_and_direction_change():
    row = _rows("mechanism_train")[0]
    residual = np.zeros((3, 26, 4), dtype=np.float64)
    residual[0, 3] = np.array([1.0, 1.0, 0.0, 0.0])
    residual[0, 4] = np.array([2.0, 1.0, 0.0, 0.0])
    residual[0, 5] = np.array([2.0, 2.0, 0.0, 0.0])
    row = replace(row, residual_features=residual)
    matrix = probes._feature_matrix((row,), "static_plus_dynamics", "target_intro_end", 5)
    assert matrix.shape == (1, 9)
    assert matrix[0, :4] == pytest.approx(residual[0, 5])
    assert matrix[0, 4:8] == pytest.approx(residual[0, 5] - residual[0, 4])
    previous_delta = residual[0, 4] - residual[0, 3]
    current_delta = residual[0, 5] - residual[0, 4]
    expected_change = 1.0 - np.dot(previous_delta, current_delta) / (
        np.linalg.norm(previous_delta) * np.linalg.norm(current_delta)
    )
    assert matrix[0, 8] == pytest.approx(expected_change)


def test_h3_h4_use_full_selection_permutations_and_h6_beats_all_registered_nulls(
    tmp_path,
):
    familiarity_selection = _selection(task="familiarity")
    answerability_selection = _selection(task="answerability")
    unsupported_selection = _selection(task="unsupported_answer")
    familiarity = _evaluate(
        tmp_path / "familiarity",
        familiarity_selection,
        _rows("probe_test", task="familiarity"),
    )
    answerability = _evaluate(
        tmp_path / "answerability",
        answerability_selection,
        _rows("probe_test", task="answerability"),
    )
    unsupported = _evaluate(
        tmp_path / "unsupported",
        unsupported_selection,
        _rows("probe_test", task="unsupported_answer"),
    )
    unsupported = replace(unsupported, relative_h6_log_loss_improvement=0.02)
    nulls = (
        _null_result(familiarity_selection, "label_permutation", 1, selected_auroc=0.2),
        _null_result(familiarity_selection, "label_permutation", 2, selected_auroc=0.8),
        _null_result(answerability_selection, "label_permutation", 1, selected_auroc=0.3),
        _null_result(answerability_selection, "label_permutation", 2, selected_auroc=0.7),
        _null_result(unsupported_selection, "layer_order", 1, h6_improvement=0.01),
        _null_result(unsupported_selection, "random_map", 1, h6_improvement=0.015),
    )
    familiarity = replace(
        familiarity,
        null_results=tuple(
            _score_null(null, familiarity, auroc=null.validation_auroc) for null in nulls[:2]
        ),
    )
    answerability = replace(
        answerability,
        null_results=tuple(
            _score_null(null, answerability, auroc=null.validation_auroc)
            for null in nulls[2:4]
        ),
    )
    unsupported = replace(
        unsupported,
        null_results=tuple(
            _score_null(
                null,
                unsupported,
                h6_improvement=null.relative_h6_log_loss_improvement,
            )
            for null in nulls[4:]
        ),
    )
    gates = evaluate_f2a_gates(familiarity, answerability, unsupported)
    assert familiarity.crossed_auroc_95 is not None
    assert answerability.crossed_auroc_95 is not None
    assert any(
        "worst reciprocal-transfer cell balanced accuracy" == criterion.name
        for criterion in gates.h3.criteria
    )
    assert isinstance(gates, F2AGates)
    assert set(gates.holm_adjusted_p) == {"H3", "H4"}
    assert gates.holm_adjusted_p["H3"] == pytest.approx(2.0 / 3.0)
    assert gates.holm_adjusted_p["H4"] == pytest.approx(2.0 / 3.0)
    assert gates.h3.criteria[-1].name == "Holm-adjusted p-value"
    assert gates.h4.criteria[-1].name == "Holm-adjusted p-value"
    assert gates.h5.criteria[0].threshold == pytest.approx(0.02)
    assert gates.h6.criteria[0].threshold == pytest.approx(0.01)
    assert gates.h6.criteria[2].name == "dynamics improvement over all layer-order nulls"
    assert gates.h6.criteria[3].name == "dynamics improvement over all random-map nulls"
    assert gates.h6.criteria[2].satisfied is True
    assert gates.h6.criteria[3].satisfied is True
    assert gates.h6_secondary is True
    assert gates.h3.status in {"supported", "not_supported", "not_evaluable"}
    with pytest.raises(TypeError):
        HypothesisGate(
            "H3",
            (GateCriterion("AUROC", 1.0, 0.65, ">="),),
            passed=True,
        )


def test_h3_h4_gates_use_registered_transfer_aggregation_not_pooled_metrics(tmp_path):
    familiarity = _evaluate(
        tmp_path / "familiarity-transfer",
        _selection(task="familiarity"),
        _rows("probe_test", task="familiarity"),
    )
    answerability = _evaluate(
        tmp_path / "answerability-transfer",
        _selection(task="answerability"),
        _rows("probe_test", task="answerability"),
    )
    unsupported = _evaluate(
        tmp_path / "unsupported-transfer",
        _selection(task="unsupported_answer"),
        _rows("probe_test", task="unsupported_answer"),
    )
    gates = evaluate_f2a_gates(familiarity, answerability, unsupported)
    assert gates.h3.criteria[0].name == "reciprocal-transfer mean AUROC"
    assert gates.h3.criteria[0].observed == familiarity.cross_condition_transfer.mean_auroc
    assert gates.h3.criteria[2].name == "worst reciprocal-transfer cell balanced accuracy"
    assert (
        gates.h3.criteria[2].observed
        == familiarity.cross_condition_transfer.worst_cell_balanced_accuracy
    )
    assert gates.h4.criteria[0].observed == answerability.cross_condition_transfer.mean_auroc

    changed_pooled = replace(
        familiarity,
        metrics=replace(familiarity.metrics, auroc=0.0, balanced_accuracy=0.0),
        worst_condition=None,
    )
    changed_gate = evaluate_f2a_gates(changed_pooled, answerability, unsupported).h3
    assert tuple((item.name, item.observed) for item in changed_gate.criteria) == tuple(
        (item.name, item.observed) for item in gates.h3.criteria
    )


def test_h3_h4_h5_gates_require_bound_pre_output_scope(tmp_path):
    results = {
        task: _evaluate(
            tmp_path / task,
            _selection(task=task),
            _rows("probe_test", task=task),
        )
        for task in probes.TASKS
    }
    with pytest.raises(ValueError, match="pre-output scope"):
        evaluate_f2a_gates(
            replace(
                results["familiarity"],
                selected_anchor="assistant_prefix_end",
                claim_scope="output_proximal_control",
            ),
            results["answerability"],
            results["unsupported_answer"],
        )
    with pytest.raises(ValueError, match="pre-output scope"):
        evaluate_f2a_gates(
            replace(results["familiarity"], selected_layer=25),
            results["answerability"],
            results["unsupported_answer"],
        )
    with pytest.raises(ValueError, match="pre-output scope"):
        evaluate_f2a_gates(
            results["familiarity"],
            results["answerability"],
            replace(
                results["unsupported_answer"],
                selected_anchor="assistant_prefix_end",
                claim_scope="output_proximal_control",
            ),
        )
    with pytest.raises(ValueError, match="assistant_prefix_end"):
        replace(
            results["familiarity"],
            selected_anchor="assistant_prefix_end",
            claim_scope="pre_output",
        )


def test_joint_gate_requires_typed_complete_null_results_and_fails_closed(tmp_path):
    selections = {
        task: _selection(task=task)
        for task in ("familiarity", "answerability", "unsupported_answer")
    }
    results = {
        task: _evaluate(
            tmp_path / task,
            selection,
            _rows("probe_test", task=task),
        )
        for task, selection in selections.items()
    }
    gates = evaluate_f2a_gates(
        results["familiarity"], results["answerability"], results["unsupported_answer"]
    )
    assert gates.h3.status == "not_evaluable"
    assert gates.h4.status == "not_evaluable"
    assert gates.h6.status == "not_evaluable"
    assert any("label-permutation" in reason for reason in gates.h3.reasons)
    assert any("layer-order" in reason for reason in gates.h6.reasons)
    with pytest.raises(TypeError):
        evaluate_f2a_gates(
            results["familiarity"],
            results["answerability"],
            results["unsupported_answer"],
            (object(),),
        )


def test_atomic_probe_bundle_evaluates_all_tasks_and_closes_once(tmp_path):
    selections = {task: _selection(task=task) for task in probes.TASKS}
    rows_by_task = {task: _rows("probe_test", task=task) for task in probes.TASKS}
    store, manifest_path, authorization = _bundle_authorization(tmp_path, selections, rows_by_task)
    bundle = evaluate_probe_bundle_once(
        selections,
        authorization,
        rows_by_task,
        store=store,
        endpoint_manifest_path=manifest_path,
    )
    assert set(bundle.results) == set(probes.TASKS)
    assert isinstance(bundle.gates, F2AGates)
    assert store.endpoint_state("probe_test", manifest_path) == "closed"
    with pytest.raises(ValueError, match="already closed"):
        evaluate_probe_bundle_once(
            selections,
            authorization,
            rows_by_task,
            store=FAArtifactStore(tmp_path),
            endpoint_manifest_path=manifest_path,
        )


def test_atomic_probe_bundle_failure_leaves_lease_resumable_not_closed(tmp_path):
    selections = {task: _selection(task=task) for task in probes.TASKS}
    rows_by_task = {task: _rows("probe_test", task=task) for task in probes.TASKS}
    store, manifest_path, authorization = _bundle_authorization(tmp_path, selections, rows_by_task)
    bad_rows = dict(rows_by_task)
    bad_rows["answerability"] = bad_rows["answerability"][1:]
    with pytest.raises(ValueError, match="sealed probe-test source identities"):
        evaluate_probe_bundle_once(
            selections,
            authorization,
            bad_rows,
            store=store,
            endpoint_manifest_path=manifest_path,
        )
    assert store.endpoint_state("probe_test", manifest_path) == "unlocked_once"
    assert not tuple((tmp_path / "runs" / "familiarity_answerability").glob("**/*metrics*"))
    bundle = evaluate_probe_bundle_once(
        selections,
        authorization,
        rows_by_task,
        store=FAArtifactStore(tmp_path),
        endpoint_manifest_path=manifest_path,
    )
    assert set(bundle.results) == set(probes.TASKS)


def test_evaluated_bundle_recovery_closes_without_rescoring_or_second_mark(
    tmp_path, monkeypatch
):
    selections = {task: _selection(task=task) for task in probes.TASKS}
    rows_by_task = {task: _rows("probe_test", task=task) for task in probes.TASKS}
    store, manifest_path, authorization = _bundle_authorization(
        tmp_path, selections, rows_by_task
    )
    original_mark = FAArtifactStore.mark_evaluated
    original_close = FAArtifactStore.close_endpoint
    calls = {"mark": 0, "close": 0}

    def tracked_mark(self, receipt, metrics_path):
        calls["mark"] += 1
        return original_mark(self, receipt, metrics_path)

    def interrupted_close(self, endpoint):
        calls["close"] += 1
        if calls["close"] == 1:
            raise RuntimeError("interrupted after mark")
        return original_close(self, endpoint)

    monkeypatch.setattr(FAArtifactStore, "mark_evaluated", tracked_mark)
    monkeypatch.setattr(FAArtifactStore, "close_endpoint", interrupted_close)
    with pytest.raises(RuntimeError, match="interrupted after mark"):
        evaluate_probe_bundle_once(
            selections,
            authorization,
            rows_by_task,
            store=store,
            endpoint_manifest_path=manifest_path,
        )
    assert store.endpoint_state("probe_test", manifest_path) == "evaluated"
    assert calls == {"mark": 1, "close": 1}

    def forbidden_recalculation(*args, **kwargs):
        raise AssertionError("evaluated recovery must not rescore protected rows")

    monkeypatch.setattr(probes, "_calculate_probe_result", forbidden_recalculation)
    recovered = evaluate_probe_bundle_once(
        selections,
        authorization,
        {},
        store=FAArtifactStore(tmp_path),
        endpoint_manifest_path=manifest_path,
    )
    assert isinstance(recovered, probes.ProbeBundleResult)
    assert store.endpoint_state("probe_test", manifest_path) == "closed"
    assert calls == {"mark": 1, "close": 2}


def test_selection_result_gates_and_bundle_have_strict_canonical_loaders(tmp_path):
    selections = {task: _selection(task=task) for task in probes.TASKS}
    rows_by_task = {task: _rows("probe_test", task=task) for task in probes.TASKS}
    store, manifest_path, authorization = _bundle_authorization(
        tmp_path, selections, rows_by_task
    )
    bundle = evaluate_probe_bundle_once(
        selections,
        authorization,
        rows_by_task,
        store=store,
        endpoint_manifest_path=manifest_path,
    )

    loaded_selections = {
        task: SelectionManifest.from_record(selection.to_record())
        for task, selection in selections.items()
    }
    loaded_results = {
        task: probes.ProbeResult.from_record(
            bundle.results[task].to_record(), selection=loaded_selections[task]
        )
        for task in probes.TASKS
    }
    loaded_gates = F2AGates.from_record(bundle.gates.to_record(), results=loaded_results)
    loaded_bundle = probes.ProbeBundleResult.from_record(
        bundle.to_record(), selections=loaded_selections
    )
    assert loaded_bundle.to_record() == bundle.to_record()
    assert loaded_gates.to_record() == bundle.gates.to_record()

    strict_records = (
        (
            selections["familiarity"].to_record(),
            lambda record: SelectionManifest.from_record(record),
        ),
        (
            bundle.results["familiarity"].to_record(),
            lambda record: probes.ProbeResult.from_record(
                record, selection=selections["familiarity"]
            ),
        ),
        (
            bundle.gates.to_record(),
            lambda record: F2AGates.from_record(record, results=bundle.results),
        ),
        (
            bundle.to_record(),
            lambda record: probes.ProbeBundleResult.from_record(
                record, selections=selections
            ),
        ),
    )
    for original, loader in strict_records:
        unknown = dict(original)
        unknown["unexpected"] = True
        with pytest.raises(ValueError, match="schema"):
            loader(unknown)
        missing = dict(original)
        missing.pop(next(iter(missing)))
        with pytest.raises(ValueError, match="schema"):
            loader(missing)

    nonfinite = bundle.to_record()
    nonfinite["results"]["familiarity"]["metrics"]["auroc"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        probes.ProbeBundleResult.from_record(nonfinite, selections=selections)
    bad_model_hash = bundle.results["familiarity"].to_record()
    bad_model_hash["selected_model_scope"]["selected_model_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="selected model"):
        probes.ProbeResult.from_record(
            bad_model_hash, selection=selections["familiarity"]
        )
    bad_derived_status = bundle.gates.to_record()
    bad_derived_status["status"] = "forged"
    with pytest.raises(ValueError, match="canonical"):
        F2AGates.from_record(bad_derived_status, results=bundle.results)
