# G05 RoboDojo Adapter

This adapter integrates a G0.5/GalaxeaVLA source checkout with XPolicyLab.
Set `G05_ROOT` to that checkout before installation, training, or evaluation.

## Installation

```bash
cd XPolicyLab/policy/G05
export G05_ROOT=/path/to/GalaxeaVLA_github_port
# Optional: use a specific policy Python/virtualenv.
export G05_PYTHON=/path/to/python
bash install.sh
```

## Training

Default training uses G0.5 task config:

```bash
cd XPolicyLab/policy/G05
export G05_ROOT=/path/to/GalaxeaVLA_github_port
export ROBODOJO_LEROBOT_V30_ROOT=/path/to/RoboDojo_lerobot_v30_video
bash train.sh RoboDojo cotrain arx_x5 joint 0 0,1,2,3,4,5,6,7
```

`train.sh` is a thin wrapper around `${G05_ROOT}/scripts/run/finetune.sh`.
Training length and checkpoint cadence remain controlled by the selected G0.5
task config and optional Hydra overrides; this adapter does not hard-code a
checkpoint step.

## Evaluation

Set `G05_CKPT_PATH` to a G0.5 run directory or `.pt` checkpoint. Debug mode
validates websocket wiring and action schema without the simulator:

```bash
cd XPolicyLab/policy/G05
export EVAL_ENV_TYPE=debug
export G05_ROOT=/path/to/GalaxeaVLA_github_port
export G05_CKPT_PATH=/path/to/g05/run/or/checkpoints/checkpoint
bash eval.sh RoboDojo stack_bowls cotrain arx_x5 joint 0 0 0 \
  /path/to/policy/python-or-venv base
```

For simulator evaluation, unset `EVAL_ENV_TYPE` or set it to `sim` and make
sure the RoboDojo evaluator-side repo with `env_cfg/`, `scripts/`, `src/`, and
`task/` is mounted next to `XPolicyLab`.

## Checkpoint

The submitted FM-only RoboDojo ARX X5 joint checkpoint and inference assets are hosted at:

https://huggingface.co/XZHY528/g05

Download and extract `xpolicylab_g05_fm_only_checkpoint_20260724.tar`, then set `G05_CKPT_PATH` to the extracted `XPolicyLab/policy/G05/checkpoints/checkpoint` path before evaluation.

Required sidecars included in the archive:

- `.hydra/config.yaml`
- `dataset_stats.json`
- `action_tokenizer.pt`
