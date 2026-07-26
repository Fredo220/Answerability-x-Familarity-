# Familiarity vs. Answerability Architecture Map

## Purpose

This document maps the active study pipeline and its evidence boundaries. It
does not provide mechanistic evidence by itself and is not an experimental
endpoint.

## Research Flow

```mermaid
flowchart LR
    A["Versioned entity candidates"] --> B["Pinned-model recall screening"]
    B --> C{"Exactly two qualified pairs per domain?"}
    C -- "No" --> D["Stop and preregister a new development amendment"]
    C -- "Yes" --> E["Tokenizer- and surface-matched pseudonyms"]
    E --> F["Two independent naturalness raters"]
    F --> G{"Naturalness gate passes?"}
    G -- "No" --> D
    G -- "Yes" --> H["F1 behavioral study on pinned Gemma"]
    H --> I{"F1 confirmatory gate passes?"}
    I -- "No" --> J["Report a negative or inconclusive result"]
    I -- "Yes" --> K["F2A activation extraction and probes"]
    K --> L["Static, dynamics, and J-space controls"]
    L --> M["Fellowship report with bounded claims"]
```

## Claim Boundary

The confirmatory question is whether entity familiarity changes answer attempts
when the requested relation is absent from context, after controlling for
answerability and surface form. F2A may test whether internal activations contain
incremental predictive information about that behavior.

The study does not claim to measure general truth, metacognition, artificial
intuition, or a universal hallucination mechanism.
