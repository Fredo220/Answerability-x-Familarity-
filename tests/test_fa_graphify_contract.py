from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_graphify_integration_is_pinned_local_and_fa_scoped():
    build_script = (REPO_ROOT / "tools" / "build_fa_graph.sh").read_text(
        encoding="utf-8"
    )
    ignore = (REPO_ROOT / ".graphifyignore").read_text(encoding="utf-8")
    architecture = (
        REPO_ROOT / "docs" / "familiarity_answerability_architecture.md"
    ).read_text(encoding="utf-8")

    assert "graphifyy==0.9.25" in build_script
    assert "--code-only" in build_script
    assert "src/trajectory_extractor/fa_*.py" in ignore
    assert "tests/test_fa_*.py" in ignore
    assert "RLMF" in architecture
    assert "does not provide mechanistic evidence" in architecture
