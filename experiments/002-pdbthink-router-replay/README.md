# Snowball RL retry with rollout routing replay

This experiment restarts PDBThink RL from the original Snowball checkpoint, with
fresh optimizer state. It uses 48 CoreWeave H100 GPUs: 32 for Megatron training
(TP1, PP2, EP16) and 16 for two vLLM rollout replicas (each DP8, EP8).

## Decisions

Enable **rollout routing replay**. The preceding controlled diagnostic reduced
the mean learner–rollout log-probability discrepancy by 65–71% when the learner
used the rollout engine's selected experts. This establishes a contribution to
probability mismatch, not a demonstrated cause of the accuracy regression.
Replay covers the positions that predict generated tokens, including the final
prompt token. Earlier prompt positions use native routing. Combine weights and
router gradients remain native.

Add **token-level truncated importance sampling**, with upper bound 4 and no
lower bound. For each valid token, the multiplier is
`min(4, exp(log p_old_learner - log p_rollout))`, detached from gradients.
MarinSkyRL multiplies this into its existing PPO/GRPO policy loss; its PPO ratio
still compares the current learner with the recomputed old learner. With one
optimizer update per batch, that latter ratio is approximately one. This is
TIS plus the existing objective, not a claim to have implemented CISPO or score
centering. It corrects residual sampling mismatch while bounding its upward
gradient multiplier. No token is discarded solely for exceeding the cap.
Simply selecting this pinned trainer's CISPO loss would still use the old
learner as its ratio denominator. The added TIS factor explicitly compares
learner probabilities with the probabilities used to generate the rollouts.

The [mismatch article](https://kiddyboots216.github.io/mismatch/) motivates
bounded correction and warns that expert IDs alone cannot eliminate differences
in routing weights or kernels. We defer its full kernel alignment and replay of
routing weights: both require broader backend changes, and replaying weights
changes the treatment of router gradients. We make no zero-mismatch claim.

[Score centering](https://arxiv.org/abs/2609.20807) is promising but deferred.
The paper subtracts an expected-score drift term, approximated using a saved
top-token distribution. That implementation is absent from upstream main
`82396a7` inspected on October 21, 2026. The unmerged
[draft implementation](https://github.com/marin-community/MarinSkyRL/pull/968)
reports CPU/GPU qualification but no established learning benefit. Adopting and
qualifying its distribution-capture and gradient path in this pinned experiment
would add a substantial independent change. The paper also studies TIS and identifies limitations of
its deliberately amplified mismatch experiments.

**Quantile/query-bias balancing remains frozen.** The original resolved config
already had `trainer.policy.grug_query_bias_update_mode: frozen`; the Grug
router uses a persistent FP32 bias and no auxiliary load-balancing loss.
Adaptive quantile updates therefore do not explain that run. This retry checks
that local bias buffers stay byte-identical across every optimizer update and
checks a full-model bias digest periodically.

## Training protocol and monitoring

- Original model revision `cfc1d845dae89b067cdc7250d0164abefa5a69cf`.
- Same frozen split: 28,045 training tasks; all 1,003 validation tasks from step
  zero, with no validation membership changes. The test split remains unused.
- Seed 17, 32 prompts/update, four responses/prompt, 876 planned updates.
- Learning rate 1e-6; one optimizer pass; gradient norm cap 1.
- Training temperature 1, top-p 1, top-k disabled, min-p 0; 8,192 response-token
  cap and 32,768 total context. Greedy validation uses remaining native context.
- Both original Snowball stop IDs (128001, 128009) are supplied explicitly in
  training and evaluation requests. Importance correction forces vLLM's neutral
  generation config, so checkpoint generation defaults cannot supply them.
- Training now explicitly uses `min_tokens=0`, as required by the native
  behavior-probability check. The previous transport default was 1, which
  suppressed EOS on the first generated token. Removing that processor makes
  the sampling distribution compatible with temperature-only learner scoring.
  Greedy validation retains its previous minimum-token default.
- Validation and checkpoint retention every 32 steps. Full trajectories retained.
- Existing per-family rewards, validation success, probability-ratio tails,
  effective sample size, and correction clipping metrics remain enabled.
- Startup independently checks installed learner/serving parameter equality and
  full-FP32 router equality before training. Missing replay data, invalid replay
  geometry, undrained recomputation routes, changed bias, nonfinite metrics, or
  nonzero rollout staleness fail the run.

The original optimizer moments are not carried forward. Because this retry
changes both replay and importance correction, a gain would measure their
combined effect. It would not isolate either component's causal contribution.

## Runtime and reproduction

Retain MarinSkyRL `5663f6129f53bb6aaa09549fa6cb423571503e94` and the original
dependency lock. Backport these upstream fixes, with source checksums:

- `b1264f778cc69fd0bfd6a0863d0775da7c0b7daa`: prediction-position alignment
  through capture, retries, chat normalization and trainer replay.
- `81ae0ef7033c500207b191a2ae7c3806f6dc78ff`: only queue recomputation routes
  when activation checkpointing actually recomputes the routers.

`prepare.py --hardware H100` creates the immutable overlay and launch config.
`overlay-manifest.json` records every modified source checksum. `launch.yaml`
contains the resolved experiment overrides; `design.json` records the decisions.
The shared experiment adapter installs the new files inside the job's runtime,
without modifying shared package caches. `submit.py` verifies the uploaded
overlay and starts a detached, durable Marin/Iris launcher.
The overlay also declares `stop_token_ids` in the sampling configuration schema;
launcher composition declares the same fields in memory. The existing native
request conversion and chat transport carry these IDs to vLLM.
The shared engine is configured without global sampling overrides, and its
wrapper honors the per-request sampling settings when those overrides are
empty. Greedy validation requests no unused rollout probabilities; the training
probability guard remains enabled. Validation errors or incomplete task coverage
stop the run.

CPU validation passed 268 upstream checks across route capture, prediction
alignment, retries, transport, replay control, sampling, importance correction
and full objective gradients, plus five experiment-specific checks of stop IDs,
training/evaluation request settings and rejection of invalid validation results.
The native launcher accepted the configuration and verified its runtime bundle.
GPU startup and optimizer-step results are recorded separately; these CPU
checks are not evidence that model training has succeeded.

Run artifacts use
`s3://marin-us-east-02a/marin/bizon/snowball-pdbthink/replay/2026.10.21-v4`.
The audit stores startup weight checks, per-step metrics and validation results.
Run `status.py` to download the audit and summarize completed optimizer steps,
probability mismatch, correction clipping and validation accuracy.

Submissions v1 and v2 were cancelled during model startup, before any optimizer
updates. The first removed an explicit neutral-generation-config override; the
second revealed that MarinSkyRL restores it internally for importance
correction. Version v3 preserves termination using explicit stop-token IDs.

Version v3 passed all-parameter equality checks on 16 rollout workers and 832
full-FP32 router comparisons with zero mismatches. It was cancelled with no
completed optimizer updates recorded after its initial evaluation exposed the
shared wrapper's sampling override. Its all-zero validation scores are execution
failures and must not be used as model measurements. The corrected v4 job is
`/bizon/snowball-pdbthink-replay-20261021-v4`.

## Correction to previous validation descriptions

The pinned serving entrypoint passes training sampling settings to the shared
OpenAI wrapper. That wrapper unconditionally overwrote each chat request's
temperature, including validation requests. Therefore, the previous native
in-training validation was configured with temperature 0 but executed with
temperature 1. Earlier descriptions of those measurements as greedy were
incorrect. This finding follows the executed source path, not just the declared
configuration.

Version v4 honors the validation request's actual temperature 0. Its accuracy
curve should be compared against its own fixed step-zero baseline; the old
in-training curve used a different sampling protocol. The separately run
original PDBThink benchmark uses a direct deployed endpoint and its frozen
temperature-0.6 requests, outside this wrapper. This finding does not change
those benchmark results. The separate full-validation baseline also used its
own direct endpoint.
