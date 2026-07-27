# Mechanistic Localization of a Systematic No-Bias in Llama 3.1 8B Instruct

Senior thesis project applying mechanistic interpretability methods to a systematic behavioral failure in Llama 3.1 8B Instruct on the ETHICS deontology benchmark.

**Author:** Quinn Potter

**Paper:** [paper.pdf](Paper.pdf)

## Summary

Llama 3.1 8B Instruct exhibits a systematic tendency to refuse valid excuses on the ETHICS deontology benchmark — label=1 accuracy of ~45% on the request subtask compared to ~91% on label=0 cases. Using attribution patching, activation patching, and mean ablation on 171 token-aligned counterfactual prompt pairs, I localize this "No-bias" to the MLP at layer 30 of the model's 32 transformer blocks.

## Key findings

- Ablating the L30 MLP at the final token position flips **16.9% of incorrect No verdicts to correct Yes**, with zero counter-directional flips.
- Yes-confidence on already-correct items increases by **Δld = +2.21** under the same intervention.
- Structurally matched control conditions produce effects roughly an order of magnitude smaller, with symmetric flip patterns that appear random.
- Attribution patching identified the L29 MLP as the strongest negative-attribution component, but causal ablation shows its effect statistically indistinguishable from zero. This attribution-ablation mismatch is consistent with known failure modes of first-order attribution methods (Kramár et al., 2024) and supports a two-stage protocol using attribution for candidate identification and causal intervention for verification.
- The L30 effect is best characterized as a content-modulated No signal over-applied at the verdict stage, rather than a constant additive bias.

## Methods

- **Model:** Llama 3.1 8B Instruct, hooked via [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
- **Dataset:** ETHICS deontology benchmark (Hendrycks et al., 2021), request subtask
- **Techniques:** attribution patching, activation patching, mean ablation
- **Counterfactuals:** 171 token-aligned clean/corrupted prompt pairs, generated with Claude Haiku 4.5 and verified for verdict flip

## Status

This is a completed senior project (Spring 2026). Appendices (full attribution rankings, per-condition ablation tables, sample counterfactuals, prompt templates) are in preparation.
