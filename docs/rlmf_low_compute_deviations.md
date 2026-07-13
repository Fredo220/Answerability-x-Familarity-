# RLMF Low-Compute Deviations

Date frozen: 2026-07-13

This study is a resource-scaled reproduction, not an exact replication. The following deviations from the source-paper-scale setup are part of the result and cannot be changed from test outcomes.

| Area | Frozen resource-scaled choice | Difference from paper-scale method |
| --- | --- | --- |
| Base model | `Qwen/Qwen3-0.6B` pinned at `c1899de289a04d12100db370d81485cdf75e47ca` | 0.6B model instead of the source-scale model |
| Dataset | 896 subject-and-answer-disjoint PopQA rows | Reduced, frozen one-task dataset |
| Training groups | Four generations with leave-one-out agreement | Training group size four |
| Correctness/equivalence | Audited short-answer proxy judge and aliases | Proxy judging rather than the source evaluation stack |
| Adaptation | LoRA rank eight, alpha 16, NF4 | Lower-rank parameter-efficient adaptation |
| Seeds | Three paired seeds: 11, 22, 33 | Finite registered seed set rather than a population claim |
| RL budget | 200 optimizer steps, checkpoint every 25 | Reduced training duration |
| Pre-SFT | 256 rows, four auxiliaries, five epochs | Five answer samples per row instead of the source's larger construction |
| Evaluation | One designated response plus 20 auxiliaries | Reduced but independent evaluation sampling budget |
| Local execution | Sequential single-checkpoint, last-token-only extraction on an 8 GB Mac | No full-state retention or simultaneous checkpoint loading |

The main-method elements retained are a separate online metacognition query for every completion in both arms and 20 independent evaluation auxiliaries per designated response. The metacognition query is bound to the exact pinned upstream prompt template in `third_party/rlmf/metacognition_prompt.txt`; only its output schema is made strict for this frozen study. Both arms run and persist it, while only RLMF uses the score for above-mean faithfulness advantage scaling.

The primary endpoint remains upstream cMFG*, not the four-sample training proxy. Study 2's primary anchor is `pre_confidence`, and its primary metric is the RLMF-minus-standard difference in incremental test-MAE gain from dynamics over static features. Any comparison with paper-scale results must present these deviations rather than imply equal compute, model scale, data volume, or judgment procedure.
