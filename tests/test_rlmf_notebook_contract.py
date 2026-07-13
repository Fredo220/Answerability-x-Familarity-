import ast
import copy
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest


NOTEBOOK_PATH = Path("notebooks/05_rlmf_colab.ipynb")
DIRECT_REQUIREMENTS = Path("requirements-rlmf-colab.txt")
COLAB_CONSTRAINTS = Path("requirements-rlmf-colab.constraints.txt")
EXPECTED_DIRECT = {
    "accelerate": "1.12.0",
    "bitsandbytes": "0.48.2",
    "datasets": "4.3.0",
    "numpy": "1.26.4",
    "pandas": "2.2.3",
    "peft": "0.18.0",
    "scikit-learn": "1.4.2",
    "scipy": "1.12.0",
    "torch": "2.7.1",
    "transformers": "4.57.1",
    "trl": "0.23.0",
}
REQUIRED_LOCKED_TOOLING = {"pip", "pytest", "setuptools", "wheel"}
REQUIRED_TRANSITIVES = {
    "huggingface-hub",
    "jinja2",
    "packaging",
    "pyarrow",
    "safetensors",
    "tokenizers",
}


def _notebook():
    return json.loads(NOTEBOOK_PATH.read_text())


def _code_cells(notebook):
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


def _tree(notebook):
    return ast.parse("\n\n".join(_code_cells(notebook)), filename=str(NOTEBOOK_PATH))


def _argv_nodes(call):
    if not call.args or not isinstance(call.args[0], (ast.List, ast.Tuple)):
        return ()
    return tuple(call.args[0].elts)


def _argv_strings(call):
    return tuple(
        node.value for node in _argv_nodes(call) if isinstance(node, ast.Constant) and isinstance(node.value, str)
    )


def _is_call(node, *parts):
    if not isinstance(node, ast.Call):
        return False
    value = node.func
    names = []
    while isinstance(value, ast.Attribute):
        names.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        names.append(value.id)
    return tuple(reversed(names))[-len(parts) :] == parts


def _assigned_value(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                return node.value
    raise AssertionError(f"missing assignment: {name}")


def _dict_value(node, key):
    assert isinstance(node, ast.Dict)
    for key_node, value_node in zip(node.keys, node.values):
        if isinstance(key_node, ast.Constant) and key_node.value == key:
            return value_node
    raise AssertionError(f"missing dictionary key: {key}")


def _literal_pins(path):
    pins = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+!-]+)", line)
        assert match is not None, f"requirement is not an exact literal pin: {line}"
        name = match.group(1).lower().replace("_", "-")
        assert name not in pins, f"duplicate requirement: {name}"
        pins[name] = match.group(2)
    return pins


class _DriveMountVisitor(ast.NodeVisitor):
    def __init__(self):
        self.guards = []
        self.mount_guards = []

    def visit_If(self, node):
        guarded = isinstance(node.test, ast.Name) and node.test.id == "USE_DRIVE"
        self.guards.append(guarded or any(self.guards))
        for child in node.body:
            self.visit(child)
        self.guards.pop()
        for child in node.orelse:
            self.visit(child)

    def visit_Call(self, node):
        if _is_call(node, "drive", "mount"):
            self.mount_guards.append(any(self.guards))
        self.generic_visit(node)


def _assert_exact_checkout(tree):
    checkout_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and {"git", "checkout", "--detach"}.issubset(_argv_strings(node))
        and any(isinstance(arg, ast.Name) and arg.id == "PROJECT_COMMIT" for arg in _argv_nodes(node))
    ]
    assert len(checkout_calls) == 1, "exact detached checkout is required"
    assert any(
        keyword.arg == "check" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True
        for keyword in checkout_calls[0].keywords
    ), "checkout must fail closed"

    resolved = _assigned_value(tree, "resolved_commit")
    revision_calls = [
        node
        for node in ast.walk(resolved)
        if _is_call(node, "subprocess", "check_output")
        and {"git", "rev-parse", "HEAD"}.issubset(_argv_strings(node))
    ]
    assert len(revision_calls) == 1
    exact_guards = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        compare = node.test
        names = {value.id for value in (compare.left, *compare.comparators) if isinstance(value, ast.Name)}
        if names == {"PROJECT_COMMIT", "resolved_commit"} and any(
            isinstance(operator, ast.NotEq) for operator in compare.ops
        ):
            exact_guards.append(any(isinstance(child, ast.Raise) for child in ast.walk(node)))
    assert exact_guards == [True], "resolved commit must be compared exactly and rejected on mismatch"


def _assert_frozen_install(tree):
    pip_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and {"pip", "install"}.issubset(_argv_strings(node))
    ]
    assert len(pip_calls) == 2, "only the frozen runtime and editable project installs are allowed"
    runtime_calls = [
        call
        for call in pip_calls
        if {
            "--constraint",
            str(COLAB_CONSTRAINTS),
            "--requirement",
            str(DIRECT_REQUIREMENTS),
        }.issubset(_argv_strings(call))
    ]
    editable_calls = [
        call for call in pip_calls if {"-e", ".", "--no-deps"}.issubset(_argv_strings(call))
    ]
    assert len(runtime_calls) == 1, "runtime install must use direct pins plus the Colab constraints"
    assert len(editable_calls) == 1, "project install must be editable with --no-deps"
    assert all(
        any(
            keyword.arg == "check"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in call.keywords
        )
        for call in pip_calls
    ), "all installs must fail closed"
    tooling = _assigned_value(tree, "TOOLING_REQUIREMENTS")
    assert isinstance(tooling, (ast.List, ast.Tuple))
    tooling_pins = {}
    for item in tooling.elts:
        assert isinstance(item, ast.Constant) and isinstance(item.value, str)
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+!-]+)", item.value)
        assert match is not None, f"tooling is not pinned exactly: {item.value}"
        tooling_pins[match.group(1).lower().replace("_", "-")] = match.group(2)
    assert tooling_pins.keys() == REQUIRED_LOCKED_TOOLING


def _assert_drive_guard(tree):
    visitor = _DriveMountVisitor()
    visitor.visit(tree)
    assert visitor.mount_guards == [True], "Drive may only be mounted under if USE_DRIVE"


def _assert_gpu_floor(tree):
    floor = _assigned_value(tree, "MIN_GPU_MEMORY_GB")
    assert isinstance(floor, ast.Constant) and floor.value == 14, "the GPU floor must be exactly 14 GB"
    guards = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        compare = node.test
        if (
            isinstance(compare.left, ast.Name)
            and compare.left.id == "gpu_memory_gb"
            and len(compare.ops) == 1
            and isinstance(compare.ops[0], ast.Lt)
            and len(compare.comparators) == 1
            and isinstance(compare.comparators[0], ast.Name)
            and compare.comparators[0].id == "MIN_GPU_MEMORY_GB"
        ):
            guards.append(any(isinstance(child, ast.Raise) for child in ast.walk(node)))
    assert guards == [True], "GPU memory below the floor must stop execution"


def _assert_sealed_inputs(tree):
    upstream_checks = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and "tests/test_rlmf_preregistration.py" in _argv_strings(node)
        and any(
            keyword.arg == "check"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in node.keywords
        )
    ]
    endpoint_checks = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _is_call(node, "store", "verify_endpoint")
        and any(isinstance(arg, ast.Constant) and arg.value == "prepare-data" for arg in node.args)
    ]
    assert len(upstream_checks) == 1, "vendored upstream manifests must be verified fail-closed"
    assert len(endpoint_checks) == 1, "the sealed prepare-data endpoint must be verified"


def _assert_registered_training_order(tree):
    stages = _assigned_value(tree, "TRAINING_STAGES")
    assert isinstance(stages, (ast.List, ast.Tuple))
    arms = []
    stage_names = []
    for stage in stages.elts:
        assert isinstance(stage, ast.Dict)
        stage_name = _dict_value(stage, "name")
        assert isinstance(stage_name, ast.Constant)
        stage_names.append(stage_name.value)
        cli = _dict_value(stage, "cli")
        values = [value.value for value in cli.elts if isinstance(value, ast.Constant)]
        if "--arm" in values:
            arms.append(values[values.index("--arm") + 1])
    assert stage_names == ["pre_sft", "standard", "rlmf"]
    assert arms == ["standard", "rlmf"], "paired arms must run standard before RLMF"

    pilot_steps = _assigned_value(tree, "PILOT_STEPS")
    assert isinstance(pilot_steps, ast.Constant) and pilot_steps.value == 25
    pilot_blocks = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "RUN_MODE"
        and any(isinstance(value, ast.Constant) and value.value == "pilot" for value in node.test.comparators)
    ]
    pilot_source = "\n".join(ast.unparse(node) for node in pilot_blocks)
    assert "--infrastructure-pilot" in pilot_source
    assert "--stop-after-step" in pilot_source
    assert "infrastructure_pilot_incomplete" in pilot_source
    assert "record.completed" in pilot_source and "record.global_step" in pilot_source


def _assert_checkpoint_archiving(tree):
    function_names = {
        node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {
        "load_checkpoint_records",
        "publish_checkpoint_archive",
        "restore_stage_archives",
        "verify_checkpoint_archive",
    }.issubset(function_names)
    assert any(_is_call(node, "subprocess", "Popen") for node in ast.walk(tree))
    assert any(
        isinstance(node, ast.While)
        and any(_is_call(child, "process", "poll") for child in ast.walk(node.test))
        and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "publish_checkpoint_archive"
            for child in ast.walk(node)
        )
        for node in ast.walk(tree)
    ), "newly sealed checkpoints must be exported while training is running"

    archive_name = _assigned_value(tree, "archive_name")
    archive_source = ast.unparse(archive_name)
    assert "record.global_step" in archive_source and "record.checkpoint_hash" in archive_source
    assert any(_is_call(node, "latest_verified_checkpoint") for node in ast.walk(tree))
    assert any(
        _is_call(node, "import_checkpoint")
        and {keyword.arg for keyword in node.keywords}
        == {
            "config",
            "expected_arm",
            "expected_pre_sft_hash",
            "expected_seed",
            "expected_stage",
        }
        for node in ast.walk(tree)
    ), "existing archives must be verified against all Task 7 bindings"
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "open"
        and any(isinstance(arg, ast.Constant) and arg.value == "xb" for arg in node.args)
        for node in ast.walk(tree)
    ), "archive publication must use exclusive creation"
    assert not any(_is_call(node, "os", "replace") for node in ast.walk(tree))


def _assert_no_hidden_scientific_logic(tree):
    forbidden_names = {"advantage", "advantages", "evaluation", "logits", "loss", "optimizer", "reward", "rewards"}
    forbidden_calls = {"backward", "evaluate", "generate", "mean", "step"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            assert node.id.lower() not in forbidden_names, f"hidden scientific variable: {node.id}"
        if isinstance(node, ast.Attribute):
            assert node.attr.lower() not in forbidden_calls, f"hidden scientific call: {node.attr}"
        if isinstance(node, ast.ImportFrom):
            assert not any(alias.name.startswith("rlmf_evaluation") for alias in node.names)


def _assert_evaluation_unavailable(notebook, tree):
    code = "\n".join(_code_cells(notebook))
    assert "RUN_EVALUATION" not in code
    assert "rlmf-generate-rollouts" not in code
    evaluation_state = _assigned_value(tree, "EVALUATION_STATE")
    status = _dict_value(evaluation_state, "status")
    owner = _dict_value(evaluation_state, "owner")
    assert isinstance(status, ast.Constant) and status.value == "not_available"
    assert isinstance(owner, ast.Constant) and owner.value == "task_10"


def _assert_final_summary(notebook):
    final_source = "".join(notebook["cells"][-1]["source"])
    final_tree = ast.parse(final_source)
    assert len(final_tree.body) == 2
    assert isinstance(final_tree.body[0], ast.Assign)
    assert len(final_tree.body[0].targets) == 1
    assert isinstance(final_tree.body[0].targets[0], ast.Name)
    assert final_tree.body[0].targets[0].id == "completion_summary"
    summary = final_tree.body[0].value
    assert isinstance(summary, ast.Dict)
    assert {
        key.value for key in summary.keys if isinstance(key, ast.Constant)
    } == {
        "artifact_root",
        "checkpoints",
        "config",
        "drive_enabled",
        "evaluation",
        "project_commit",
        "run_mode",
        "run_state",
        "schema_version",
        "seed",
    }
    assert isinstance(final_tree.body[1], ast.Expr)
    assert any(_is_call(node, "json", "dumps") for node in ast.walk(final_tree.body[1]))
    assert not any(
        word in final_source.lower()
        for word in ("better", "conclusion", "improved", "significant", "successful model")
    )


def validate_notebook_contract(notebook):
    assert notebook["nbformat"] == 4
    code_cells = _code_cells(notebook)
    assert code_cells
    for index, source in enumerate(code_cells):
        compile(source, f"{NOTEBOOK_PATH}:cell-{index}", "exec")
    tree = _tree(notebook)
    _assert_exact_checkout(tree)
    _assert_frozen_install(tree)
    _assert_drive_guard(tree)
    _assert_gpu_floor(tree)
    _assert_sealed_inputs(tree)
    _assert_registered_training_order(tree)
    _assert_checkpoint_archiving(tree)
    _assert_no_hidden_scientific_logic(tree)
    _assert_evaluation_unavailable(notebook, tree)
    _assert_final_summary(notebook)


def test_notebook_satisfies_executable_task_8_contract():
    validate_notebook_contract(_notebook())


def test_colab_direct_and_transitive_dependencies_are_frozen_exactly():
    assert COLAB_CONSTRAINTS.is_file(), "checked-in Linux/Colab constraints are required"
    direct = _literal_pins(DIRECT_REQUIREMENTS)
    constrained = _literal_pins(COLAB_CONSTRAINTS)
    assert direct == EXPECTED_DIRECT
    assert direct.items() <= constrained.items()
    assert REQUIRED_LOCKED_TOOLING <= constrained.keys()
    assert REQUIRED_TRANSITIVES <= constrained.keys()
    header = COLAB_CONSTRAINTS.read_text().splitlines()[:8]
    assert any("Python 3.12" in line and "Linux x86_64" in line for line in header)
    assert any("uv pip compile" in line for line in header)


def test_restore_verifies_same_step_archives_and_imports_completed_canonical_only():
    helper_cells = [
        source
        for source in _code_cells(_notebook())
        if "def restore_stage_archives" in source
    ]
    assert len(helper_cells) == 1
    commands = []
    archives = [Path("/drive/incomplete.tar"), Path("/drive/completed.tar")]
    namespace = {
        "ARTIFACT_ROOT": Path("/content/rlmf-scratch/artifacts"),
        "CONFIG_PATH": "configs/rlmf_qwen06b_confirmatory.json",
        "DRIVE_EXPORT_ROOT": SimpleNamespace(glob=lambda pattern: archives),
        "Path": Path,
        "USE_DRIVE": True,
        "config": SimpleNamespace(study_id="study"),
        "subprocess": SimpleNamespace(run=lambda command, check: commands.append((command, check))),
    }
    exec(compile(helper_cells[0], "task-8-helper-cell", "exec"), namespace)
    records = {
        archives[0]: SimpleNamespace(
            checkpoint_hash="a" * 64, completed=False, global_step=25, micro_step=100
        ),
        archives[1]: SimpleNamespace(
            checkpoint_hash="b" * 64, completed=True, global_step=25, micro_step=100
        ),
    }
    verified = []
    namespace["verify_checkpoint_archive"] = lambda path, stage, parent: (
        verified.append(path) or records[path]
    )
    namespace["load_checkpoint_records"] = lambda stage: []
    stage = {
        "name": "pre_sft",
        "record_stage": "pre_sft",
        "record_arm": None,
        "seed": None,
    }

    restored = namespace["restore_stage_archives"](stage, None)

    assert len(verified) == 2
    assert set(verified) == set(archives)
    assert restored == ["b" * 64]
    assert len(commands) == 1
    assert commands[0][1] is True
    assert str(archives[1]) in commands[0][0]


def _replace_code(notebook, old, new):
    matches = 0
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if old in source:
            matches += source.count(old)
            cell["source"] = source.replace(old, new).splitlines(keepends=True)
    assert matches == 1, f"mutation target count for {old!r}: {matches}"


def _mutate_exact_checkout(notebook):
    _replace_code(notebook, "if resolved_commit != PROJECT_COMMIT:", "if False:")


def _mutate_unpinned_install(notebook):
    notebook["cells"].append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                'subprocess.run([sys.executable, "-m", "pip", "install", "rogue-package"], check=True)\n'
            ],
        }
    )


def _mutate_drive_guard(notebook):
    _replace_code(notebook, "if USE_DRIVE:\n    from google.colab import drive", "if True:\n    from google.colab import drive")


def _mutate_gpu_floor(notebook):
    _replace_code(notebook, "MIN_GPU_MEMORY_GB = 14", "MIN_GPU_MEMORY_GB = 0")


def _mutate_sealed_verification(notebook):
    _replace_code(
        notebook,
        'data_manifest = store.verify_endpoint(config.study_id, "prepare-data")',
        'data_manifest = {"endpoint": "prepare-data", "parent_hashes": {}}',
    )


def _mutate_arm_order(notebook):
    training_cells = [
        cell
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "TRAINING_STAGES =" in "".join(cell["source"])
    ]
    assert len(training_cells) == 1
    source = "".join(training_cells[0]["source"])
    standard_pattern = r'("--arm",\s*)"standard"'
    rlmf_pattern = r'("--arm",\s*)"rlmf"'
    source, standard_count = re.subn(standard_pattern, r'\1"temporary"', source, count=1)
    source, rlmf_count = re.subn(rlmf_pattern, r'\1"standard"', source, count=1)
    source, temporary_count = re.subn(
        r'("--arm",\s*)"temporary"', r'\1"rlmf"', source, count=1
    )
    assert (standard_count, rlmf_count, temporary_count) == (1, 1, 1)
    training_cells[0]["source"] = source.splitlines(keepends=True)


def _mutate_hidden_math(notebook):
    notebook["cells"].insert(
        -1,
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["loss = logits.mean()\n", "loss.backward()\n"],
        },
    )


def _mutate_final_summary(notebook):
    notebook["cells"][-1]["source"].append('conclusion = "significant improvement"\n')


@pytest.mark.parametrize(
    "mutator",
    [
        pytest.param(_mutate_exact_checkout, id="exact-checkout-removed"),
        pytest.param(_mutate_unpinned_install, id="unpinned-install-added"),
        pytest.param(_mutate_drive_guard, id="drive-mounted-unconditionally"),
        pytest.param(_mutate_gpu_floor, id="gpu-floor-removed"),
        pytest.param(_mutate_sealed_verification, id="sealed-verification-bypassed"),
        pytest.param(_mutate_arm_order, id="paired-order-reversed"),
        pytest.param(_mutate_hidden_math, id="hidden-scientific-math-added"),
        pytest.param(_mutate_final_summary, id="interpretive-summary-added"),
    ],
)
def test_adversarial_notebook_mutations_are_rejected(mutator):
    mutated = copy.deepcopy(_notebook())
    mutator(mutated)
    with pytest.raises(AssertionError):
        validate_notebook_contract(mutated)
