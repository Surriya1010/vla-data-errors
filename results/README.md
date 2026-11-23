# Results

Nothing here yet. Each result folder must contain, or link to:

- the exact command and git commit of this repo that produced it;
- `configs/train.env` values at the time (copy the file into the folder);
- the dataset revision (commit SHA) and the condition's `manifest.json`;
- `lerobot-eval` output, including the 95% interval (`pc_success_ci95`).

Layout:

```
eval/<condition>_seed<N>/   lerobot-eval outputs
eval/official_*/            pipeline checks with published checkpoints
detectors/<condition>.csv   per-episode detector outputs
course/                     plots from the Isaac Lab course part
figures/                    final charts
```
