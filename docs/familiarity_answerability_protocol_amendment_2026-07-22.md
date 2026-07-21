# Protocol Amendment: 2026-07-22

```yaml
date: 2026-07-22
pre_outcome: true
affected_endpoints:
  - mechanism_train
  - locked_validation
  - behavior_test
  - probe_test
  - intervention_test
  - pilot
  - circuit_dev
```

## Rationale

The implementation protocol is being frozen before construction or opening of any Familiarity-vs-Answerability outcome endpoint. This amendment resolves the entity-unit allocation and source revisions required for reproducible implementation.

## Amendment

The confirmatory split counts are fixed at `mechanism_train=64`, `locked_validation=32`, `behavior_test=48`, `probe_test=24`, and `intervention_test=24`. `pilot` and `circuit_dev` are non-confirmatory namespace permissions only; they are excluded from confirmatory entity counts and claims.

The confirmatory model and tokenizer revision is `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8`; the Gemma chat-template SHA-256 is `ecd6ae513fe103f0eb62e8ab5bfa8d0fe45c1074fa398b089c93a7e70c15cfd6`; Gemma Scope is pinned at `fd571b47c1c64851e9b1989792367b9babb4af63`; and optional circuit-tracer is pinned at `4bb8c0ea10bde09727e14565ec8469656880da53` (`v0.5.0`). These pins are recorded in `data/fa/source_pins.json`; gated model access has not been tested by this amendment.

No outcomes were generated, inspected, or used to select the counts, pins, thresholds, template families, layers, directions, or claims described here.
