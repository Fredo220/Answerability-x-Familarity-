# Pilot v13 Layer Serialization Correction

**Date:** 2026-07-24
**Scope:** Familiarity versus Answerability pilot only
**Status:** Post-pilot documentation correction; confirmatory choices unchanged

The machine-readable v13 amendment serialized the registered layer list as
`[0, 18, 9, 18, 27]`, duplicating layer 18. The corresponding prose
amendment and executed analysis used the intended ordered set
`[0, 9, 18, 27]`.

The sealed v13 JSON is not edited in place. This note records the discrepancy
and preserves the original artifact hash. The duplicate did not create an
additional activation, feature, statistical comparison, or selection
opportunity in the executed pilot.

This correction is not evidence for a confirmatory claim and does not change
the pinned confirmatory layers, thresholds, model revision, or protected
endpoints.
