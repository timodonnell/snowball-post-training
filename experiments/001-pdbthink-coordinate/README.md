# PDBThink coordinate RL

[Issue #1](https://github.com/timodonnell/snowball-post-training/issues/1).
Train Snowball directly against PDBThink's native verifier with Marin's artifact
launcher, MarinSkyRL, Megatron, vLLM, and Iris on CoreWeave. No GLM responses or
other teacher traces are used.

## Status

The cohort is prepared and verified. Unit tests, the installed environment, the
Marin artifact plan, Hydra composition, and MarinSkyRL's live `prepare` preflight
pass. The verified model is staged on CoreWeave and the GPU smoke coordinator
has been submitted. Baseline measurements and optimizer steps are pending. No
improvement is claimed. See `runs.json` for the current run record.

## Frozen inputs

| Input | Revision |
| --- | --- |
| `open-athena/Snowball-67B-A2B-5.7T-Mixed-RLVR-Step38` | `cfc1d845dae89b067cdc7250d0164abefa5a69cf` |
| `open-athena/pdbthink-coordinate-tasks` v1.3.0 | `3734406cb97b1702844319f9a5d860cbbf8fe660` |
| `marin-community/marin` | `44c95a7914864fdab8b74d597052e30cbc92b8cc` |
| `marin-community/MarinSkyRL` | `5663f6129f53bb6aaa09549fa6cb423571503e94` |

`cohort_manifest.json` records the parquet, tokenizer, template, verifier, and
cohort checksums. The full ID list is in the prepared artifact's `cohort.json`.
Preparation verifies the release checksums, joins the context table on task path
and prompt hash, and re-tokenizes every selected prompt using the pinned native
template with thinking enabled. Eligible tasks have at least 8,192 tokens left
in Snowball's 32,768-token context. Published splits and source groups remain
intact and disjoint:

| Split | Tasks | Purpose |
| --- | ---: | --- |
| Train | 28,045 | RL pool |
| Validation | 1,003 | Development and checkpoint selection |
| Test | 1,370 | Reserved final comparison |
| Monitor | 107 | Fixed subset of validation, at most eight per family |
| Smoke | 32 | Training subset covering 18 families and long prompts |

Validation covers 14 families. It has no T01, S06, I01, S07, or S08 examples;
do not infer held-out performance on these families. The manifest gives exact
counts for each split and family.

Durable prepared input prefix:
`s3://marin-us-east-02a/marin/bizon/snowball-pdbthink/inputs/2026.10.07-v2`.
Model mirror prefix:
`s3://marin-us-east-02a/marin/bizon/snowball-pdbthink/models/cfc1d845dae89b067cdc7250d0164abefa5a69cf`.
The model mirror writes `source.json` only after all files pass size and SHA256
checks. The launcher requires that readiness manifest before allocating GPUs.

## Reward and evaluation

The model receives only `prompt.json`'s original system and user messages. Gold
answers are environment extras. No solution archive is read. No tools are
executed; structured requests explicitly disable tools. The pinned task's
`tests/coordinate_scoring` implementation decides exact `FINAL:` correctness,
including v1.3.0's inclusive numeric tolerance. Reward is binary correctness;
partial set scores are diagnostics only. Truncation, refusal, tool calls, and
oversized responses receive zero reward.

Training samples four responses per prompt at temperature 1.0 with an 8,192-token
output cap. **Evaluation uses the full remaining context**, `32768 - input_tokens`,
temperature 0, top-p 1, and one response per task. The 8K cohort reserve is not an
evaluation cap. Baseline and trained evaluations must use identical tasks and
this same budget policy. Raw responses, termination evidence, token counts,
format failures, and verifier diagnostics are retained.

Training logs `reward/domain/<family>/avg_raw_reward`; the fixed validation panel
logs `eval/<family>/avg_score` and `eval/<family>/pass_at_1` to W&B project
`timodonnell/snowball-pdbthink`. `monitor.py` produces task-weighted and family
macro curves with coverage. Monitor accuracy is a development signal, not the
full validation or test result. `evaluate.py` runs resumable full held-out
comparisons against an OpenAI-compatible endpoint and checks the server's prompt
token count against the frozen native count.

## Training configuration

`launch.py` is the configuration source and uses `@rl_build_options`. The explicit
role plan requests 40 H100s on `cw-rno2a`: four eight-GPU Megatron policy nodes
(TP1, PP2, EP8) plus one eight-GPU vLLM rollout node (DP8, EP8). Each node requests
64 CPUs, 1,800 GB host RAM, and 1,000 GB disk, following MarinSkyRL's Snowball
Megatron recipe. The current pinned runtime supports Megatron for this model.

GRPO uses learning rate 1e-6, 32 prompts per update, four samples per prompt,
microbatch size one, gradient checkpointing, optimizer offload during rollout,
and no KL/reference model. Smoke runs two updates; pilot runs at most 64 updates
drawn from the full training pool. The optional `epoch` scale permits 876 full
batches; upstream batching may omit the final incomplete batch. A pilot does
not imply every training task has been visited.

Validation runs before training, every eight pilot updates, and at completion.
Smoke evaluates at step two. Checkpoints follow the same interval, with two
resume checkpoints retained and a terminal Hugging Face export. Temporary
training/trajectory artifacts follow Marin's 14-day TTL; preserve any needed
raw traces before expiry. Final exports and terminal metadata use the durable
Marin artifact prefix.

`adapter.py` is a small, checksum-pinned runtime overlay because the current
MarinSkyRL environment registry has no external plugin setting. It bundles the checksum-verified native chat template, registers
`PDBThinkEnv`, and preserves an explicitly disabled tool list in structured chat
transport. Training, loss, model, placement, and checkpoint logic remain in the
pinned upstream runtime. The exact overlay SHA256 is
`8430c1b3739a4d0c66d8ec92e5e71c8fc51cf409f8afc2798e139bcae593cc29`.

## Reproduce

From the repository root, install local preparation/evaluation tools:

```bash
uv sync --extra operations
export PYTHONPATH=experiments/001-pdbthink-coordinate
hf download open-athena/pdbthink-coordinate-tasks --repo-type dataset \
  --revision 3734406cb97b1702844319f9a5d860cbbf8fe660 --local-dir data/tasks
hf download open-athena/Snowball-67B-A2B-5.7T-Mixed-RLVR-Step38 \
  --revision cfc1d845dae89b067cdc7250d0164abefa5a69cf \
  --include '*.json' '*.jinja' --local-dir data/model-metadata
uv run python -m snowball_pdbthink.prepare --dataset data/tasks \
  --model-metadata data/model-metadata --output data/pdbthink-001
uv run python -m snowball_pdbthink.adapter build --prepared data/pdbthink-001
uv run pytest -q
```

Use a pinned Marin checkout with its CPU dependencies installed. `run_marin.py`
verifies its revision and copies this experiment into its packaged workspace.
It can use an existing compatible Marin Python environment via `--python`.

```bash
git clone https://github.com/marin-community/marin.git data/marin
git -C data/marin checkout 44c95a7914864fdab8b74d597052e30cbc92b8cc
uv sync --project data/marin --extra cpu
```

Stage data using a Marin environment (which supplies `rigging`) and mirror the
model using local operations dependencies. For a shared model cache, supply
`--model-cache /path/to/cache --keep-cache`; otherwise shards are removed from
the download cache after successful upload. `coreweave.py` supplies the existing
CoreWeave task credentials to a child command without printing or saving them.

```bash
uv run python -m snowball_pdbthink.coreweave --kubeconfig "$KUBECONFIG" -- \
  data/marin/.venv/bin/python -m snowball_pdbthink.stage \
  --prepared data/pdbthink-001 \
  --data-uri s3://marin-us-east-02a/marin/bizon/snowball-pdbthink/inputs/2026.10.07-v2
uv run python -m snowball_pdbthink.coreweave --kubeconfig "$KUBECONFIG" -- \
  .venv/bin/python -m snowball_pdbthink.stage \
  --model-uri s3://marin-us-east-02a/marin/bizon/snowball-pdbthink/models/cfc1d845dae89b067cdc7250d0164abefa5a69cf
```

First plan without `--run`. Append `--run` to submit through the Marin Iris hub.
Only launch `--scale pilot` after the smoke advances optimizer steps and produces
its checkpoint/export. Use a fresh immutable calendar version for a changed run.

```bash
uv run python experiments/001-pdbthink-coordinate/run_marin.py \
  --marin data/marin --python data/marin/.venv/bin/python \
  --version 2026.10.07.4 --scale smoke \
  --data-uri s3://marin-us-east-02a/marin/bizon/snowball-pdbthink/inputs/2026.10.07-v2 \
  --model-uri s3://marin-us-east-02a/marin/bizon/snowball-pdbthink/models/cfc1d845dae89b067cdc7250d0164abefa5a69cf \
  --adapter-sha256 8430c1b3739a4d0c66d8ec92e5e71c8fc51cf409f8afc2798e139bcae593cc29
```

Inspect the coordinator and its child jobs using `iris --cluster=marin job
describe JOB_ID` and `iris --cluster=marin job logs JOB_ID`. Export the monitor
curve with the W&B run ID recorded in `runs.json`:

```bash
uv run python -m snowball_pdbthink.monitor \
  --run timodonnell/snowball-pdbthink/RUN_ID \
  --manifest data/pdbthink-001/manifest.json --output runs/001/monitor.json
uv run python -m snowball_pdbthink.evaluate \
  --tasks data/pdbthink-001/validation.parquet \
  --verifier data/pdbthink-001/native_verifier \
  --base-url http://MODEL_ENDPOINT/v1 --model SERVED_MODEL_NAME \
  --model-identity EXACT_CHECKPOINT_ID --output runs/001/validation-CHECKPOINT_ID
```

For a dedicated held-out server, use `run_marin.py --module
experiments.snowball_pdbthink.serve_evaluation --model-uri CHECKPOINT_URI --name
UNIQUE_NAME`, with the same `--marin` and `--python` arguments. This requests
eight H100s, runs the pinned Marin vLLM fork with TP1/DP8/EP8, and stops after
four hours. Its loader uses four readers and a 120-second S3 low-throughput
window, following observed model-load timeouts. These settings must match for
baseline and trained checkpoints. The base fork currently resolves to
`01911be34fac` in the pinned Marin runtime.

Use `run_marin.py --module experiments.snowball_pdbthink.evaluate_iris` with
`--endpoint /serve/UNIQUE_NAME`, absolute `--tasks`, `--verifier`, `--output`, and
`--model-identity` paths/identity. It waits for readiness and keeps its scoped
Iris capability in process memory. It never prints or saves that capability.
Stop the dedicated server after evaluation with `iris --cluster=marin job cancel
/bizon/UNIQUE_NAME`; a completed evaluator does not itself stop the server.

Use different output directories for baseline and trained checkpoints. Record
the selected checkpoint before evaluating test; report all family counts,
macro and task-weighted accuracy, coverage, formatting, truncation, and token
usage. Sparse monitor gains alone are insufficient to claim improvement.
