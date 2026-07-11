from trajectory_extractor.artifacts import RunStore
from trajectory_extractor.report_builder import write_study_report


def test_report_builder_marks_missing_stages_without_inventing_results(tmp_path):
    path = write_study_report(RunStore(tmp_path / "runs"), tmp_path / "results.md")
    text = path.read_text()
    assert "not run" in text
    assert "not completed" in text
    assert "Remizov/Chernoff is mathematical inspiration" in text
