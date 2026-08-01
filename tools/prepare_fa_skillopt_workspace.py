#!/usr/bin/env python3
"""Create the minimal workspace visible to the external SkillOpt backend."""

from __future__ import annotations

import argparse
import shlex
import shutil
from pathlib import Path


TARGET = Path(".agents/skills/fa-research-workflow/SKILL.md")
TASKS = Path("skillopt/fa_research_workflow_tasks_v1.json")
ALLOWLIST = (TARGET, TASKS)


def prepare(
    project: Path, destination: Path, allowed_root: Path, codex_path: Path | None = None
) -> Path:
    project = project.resolve()
    destination = destination.resolve()
    allowed_root = allowed_root.resolve()
    if destination == project or project in destination.parents:
        raise ValueError("sanitized workspace must be outside the source project")
    if destination == allowed_root or allowed_root not in destination.parents:
        raise ValueError("sanitized workspace must be a child of the allowed root")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("sanitized workspace must be new and empty")
    for relative in ALLOWLIST:
        source = project / relative
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"invalid allowlisted source: {relative}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (destination / "README.md").write_text(
        "# Sanitized SkillOpt workspace\n\n"
        "This generated directory contains only the reviewed workflow skill and tasks.\n",
        encoding="utf-8",
    )
    if codex_path is not None:
        codex_path = codex_path.resolve()
        profile = destination / "codex.sb"
        profile.write_text(
            '(version 1)\n(allow default)\n'
            f'(deny file-read* (subpath "{project}"))\n',
            encoding="utf-8",
        )
        wrapper = destination / "codex-sandboxed"
        wrapper.write_text(
            "#!/bin/sh\nexec /usr/bin/sandbox-exec -f "
            f"{shlex.quote(str(profile))} {shlex.quote(str(codex_path))} \"$@\"\n",
            encoding="utf-8",
        )
        wrapper.chmod(0o700)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--allowed-root", type=Path, required=True)
    parser.add_argument("--codex-path", type=Path)
    args = parser.parse_args()
    print(prepare(args.project, args.destination, args.allowed_root, args.codex_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
