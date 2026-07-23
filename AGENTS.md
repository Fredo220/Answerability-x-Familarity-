## graphify

This worktree has a Familiarity-vs-Answerability-only knowledge graph at
`graphify-out/`. The scope is defined by `.graphifyignore`; do not expand it to
archived RLMF, jailbreak, Remizov, or concept-mixing tracks.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- Use Graphify `0.9.25` through `uvx --from graphifyy==0.9.25 graphify`.
- For FA codebase questions, first run `uvx --from graphifyy==0.9.25 graphify query "<question>" --graph graphify-out/graph.json` when the graph exists. Use the pinned command with `path` or `explain` for focused relationships.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- Treat `EXTRACTED` edges as code facts and `INFERRED` edges only as hypotheses to verify against source.
- After modifying FA code, run `tools/build_fa_graph.sh --force` to keep the graph current. This is local AST extraction only and uses no external model.
