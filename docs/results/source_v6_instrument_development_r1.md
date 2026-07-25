# Source-v6 Instrument Development R1

**Status:** Failed open-development instrument revision
**Protected endpoints:** Unopened
**Construction validation:** Unopened

The pinned Gemma-2-2B run completed all 288 registered prompts for 96
`instrument_development` entities. It qualified 61 entities overall, but the
place domain qualified only 4 of 24 entities. The other domains qualified
17-21 of 24 entities.

The place shortfall is concentrated in:

- `P131` administrative region: 4/24 correct;
- `P421` time zone: 2/24 correct.

Development-only error inspection found three distinct problems:

1. the generic `P131` wording does not specify an administrative level;
2. several `P421` source values are historical, overly broad, or inconsistent
   with ordinary current time-zone answers;
3. raw completion scoring creates a smaller number of conservative parser false
   negatives.

R1 is retained as a failed instrument revision. It cannot support a
Familiarity, Answerability, hallucination, intuition, or mechanism claim. A new
revision may change only the open instrument and must pass its frozen
development gate before `construction_validation` is opened.

Machine-readable provenance and aggregate results are in
`source_v6_instrument_development_r1.json`. The complete immutable run archive is
stored in the study's private Google Drive checkpoint directory under SHA-256:

```text
1b06dc9180c060624666b3ddd60c6f4537bf318c216ec76b9f7395a39e073cc4
```
