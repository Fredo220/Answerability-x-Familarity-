# Source-v6 R8 QLever Preflight

**Status:** Passed before construction

The Git-bound R8 commit was
`0aa6f0ef0798073a0a464b3639479d41083d77ac`.
The first local attempt stopped before reaching QLever because the system
Python certificate store could not verify the endpoint certificate. The exact
queries were then run with the `certifi` CA bundle already used by the project.

The `person`, `place`, and `organization` queries returned the five registered
binding columns. The `creative_work` query returned HTTP 500; one exact
transport retry returned the same registered columns. No candidate identity,
rank, domain yield, model output, or research endpoint was retained or
inspected.

The preflight therefore passed without changing query text, relations, limits,
ordering, thresholds, or source semantics.
