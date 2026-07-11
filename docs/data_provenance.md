# Data Provenance

## Controlled concept-mixing set

`feature-dynamics generate-concept-data` creates 1,200 unique synthetic
entity-relation-object tasks. Surface entities, target facts, entity families,
and prompt-template groups are disjoint across train, validation, and test.
Every task contains a same-relation hard distractor. The generator is
deterministic from its recorded seed. Relation, distractor count, name
similarity, and rarity proxy are marginally balanced in every split; answer
position is balanced within each distractor-count condition. Factor schedules
are shuffled independently to avoid deterministic shortcuts such as target
identity or template predicting distractor count. Every fact table uses unique
entities and unique distractor objects. A sidecar manifest records generator
schema, seed, split counts, balanced factors, and the JSONL SHA-256 checksum.

## Real-transfer set

Each JSONL row must contain `id`, `subject`, `relation`, `object`, `source_url`,
`distractors`, and `distractor_answers`. Invalid URLs and files with fewer than
200 rows are rejected. `feature-dynamics prepare-real-transfer` queries
Wikidata for human-place-of-birth facts, keeps unique English labels, selects
four different-answer distractors by name similarity, and records the exact
SPARQL query plus output checksum. The result is an external transfer test,
not a source of training features or threshold selection.

## Jailbreak set

Only exports from the official JailbreakBench repository and frozen published
attack artifacts are accepted. Strict validation requires 100 harmful and 100
matched benign rows, unique IDs, named categories, official provenance, and a
frozen artifact for every harmful behavior. Every matched pair must share a
`pair_id` with exactly one harmful and one benign row. The loader does not synthesize or
mutate attacks. Raw benchmark data remains governed by its upstream license and
should be placed under `data/external/`, which is not committed.

`feature-dynamics prepare-jailbreak-data` performs the join by official index
and verifies `Behavior` and `Category` fields after case/whitespace
normalization before writing a study
file. It records SHA-256 checksums and the exact artifacts repository commit in
a sidecar manifest. The benign model input is the official `Goal`; `Behavior`
is metadata only. The initial CPU-first protocol freezes the published DSN
artifact optimized for Llama-2 7B because its prompts are short. Transfer to
Llama 3.2 1B is unknown before evaluation and is not assumed.

## Model access

Llama 3.2 1B Instruct and Llama Guard 3 1B are gated. Users must accept Meta's licenses and authenticate with Hugging Face locally. Model weights are never copied into run artifacts.
