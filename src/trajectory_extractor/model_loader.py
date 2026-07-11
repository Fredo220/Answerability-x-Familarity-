from __future__ import annotations

import gc

import torch

from trajectory_extractor.types import ExperimentConfig


DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"


def load_hf_model(config: ExperimentConfig):
    """Load one gated model at a time for the CPU-first pipeline."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Install project dependencies with `pip install -e .`.") from exc

    dtype = _resolve_dtype(config.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.model_revision,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(config.device)
    model.eval()
    return model, tokenizer


def unload_model(model) -> None:
    """Release the target before loading Llama Guard on an 8 GB machine."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc
