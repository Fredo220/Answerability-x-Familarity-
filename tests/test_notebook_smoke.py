import json
from pathlib import Path


def test_study_notebooks_import_package_and_consume_artifacts():
    paths = sorted(Path("notebooks").glob("*.ipynb"))
    assert [path.name for path in paths] == [
        "01_sanity_check.ipynb",
        "02_concept_mixing.ipynb",
        "03_jailbreak.ipynb",
        "04_intervention.ipynb",
    ]
    for path in paths:
        notebook = json.loads(path.read_text())
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        assert "trajectory_extractor" in source
    assert "RunStore" in json.dumps(json.loads(paths[0].read_text()))
