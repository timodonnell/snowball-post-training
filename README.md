# Snowball post-training

Experiments for post-training Snowball, with reproducible configs and evaluation results.

Current experiment: [PDBThink coordinate RL](experiments/001-pdbthink-coordinate/README.md),
using MarinSkyRL on CoreWeave and the 28,045-task native-context training cohort.

- [`experiments/`](experiments/): one directory per experiment, containing its
  code, configs, commands and result summary.
- [GitHub issues](https://github.com/timodonnell/snowball-post-training/issues):
  experiment goals, scope and progress.

Pin model and dataset revisions. Keep datasets, checkpoints and detailed run logs
outside Git; `data/`, `checkpoints/` and `runs/` are ignored.
