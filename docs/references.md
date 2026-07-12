# Research Basis

- [Azaria and Mitchell, The Internal State of an LLM Knows When It's Lying](https://arxiv.org/abs/2304.13734)
- [Marks and Tegmark, The Geometry of Truth](https://arxiv.org/abs/2310.06824)
- [Galkin and Remizov, Upper and lower estimates for rate of convergence in the Chernoff product formula for semigroups of operators](https://arxiv.org/abs/2104.01249)
- [Toy Models of Superposition](https://www.transformer-circuits.pub/2022/toy_model/index.html)
- [Towards Monosemanticity](https://www.transformer-circuits.pub/2023/monosemantic-features/index.html)
- [JailbreakBench](https://arxiv.org/abs/2404.01318)
- [Inference-Time Intervention](https://arxiv.org/abs/2306.03341)
- [Anthropic, Open-sourcing circuit tracing tools](https://www.anthropic.com/research/open-source-circuit-tracing)
- [Decode Research, circuit-tracer](https://github.com/decoderesearch/circuit-tracer)
- [Hugging Face Llama model API](https://huggingface.co/docs/transformers/main/model_doc/llama)
- [Anthropic, Persona Vectors](https://www.anthropic.com/research/persona-vectors)
- [Anthropic, The Assistant Axis](https://www.anthropic.com/research/assistant-axis)
- [Anthropic, Emotion Concepts and their Function in a Large Language Model](https://transformer-circuits.pub/2026/emotions/index.html)

These works motivate the baselines and experimental questions. They do not establish that transformer layers form a Chernoff family or that PCA features are monosemantic.

Persona Vectors, the Assistant Axis, and the emotion-concepts work establish that
distributed activation directions can monitor and causally affect behavior. This
project therefore does not claim the first activation-vector monitor, dynamic
internal monitor, or safety steering method. Its narrower registered question is
whether causal layer/token dynamics of a contrastive direction add held-out value
for entity-relation-object binding failures in a small open model.
