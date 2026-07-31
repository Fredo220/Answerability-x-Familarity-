# Same-String Primary Study Amendment

**Date:** 2026-08-01
**Study ID:** `familiarity-answerability-same-string-gemma2-2b-v1`
**Run ID:** `same-string-primary-v1`
**Status:** Registered pre-outcome

## Scope

This amendment registers an isolated Same-String primary study using the
existing confirmatory Gemma model, tokenizer, chat-template, generation,
split, bootstrap, anchor, and threshold values. Its SHA-256 hash is the
amendment hash bound into later sealed Same-String artifacts.

## Preserved R11 record

R11 is immutable and remains `not_evaluable`. This study does not repair,
reinterpret, replace, or reopen any R11 artifact or endpoint.

## Primary estimand

H2b becomes the primary Same-String behavioral estimand: the registered
difference-in-differences in answer attempts between high- and low-exposure
conditions when the archive code is absent, net of the same exposure contrast
when the code is target-bound. The registered H2b effect, bootstrap, output
validity, complete-cell, and capability-preservation gates remain unchanged.

## Gated follow-up

All mechanistic work is gated behind completion of the registered behavioral
study and its behavioral gates. All causal work is additionally gated behind
the registered behavioral and probe gates. These follow-ups cannot alter or
rescue the primary Same-String result.

## Identity boundary

Only the `study_id` and `run_id` differ from the existing confirmatory Gemma
configuration. This amendment changes no R11 artifact and no code behavior.
