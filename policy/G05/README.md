# G05 RoboDojo Adapter

This directory contains the XPolicyLab adapter for evaluating and reproducing a
G0.5/GalaxeaVLA policy on RoboDojo.

The adapter intentionally does not vendor the GalaxeaVLA source tree. Set
`G05_ROOT` to a local GalaxeaVLA checkout before installing, training, or
evaluating.

## File Structure

| Path | Purpose |
|---|---|
| `README.md` | Usage notes for the G05 RoboDojo adapter. |
| `install.sh` | Installs XPolicyLab websocket/server dependencies and, when possible, the GalaxeaVLA checkout. |
| `train.sh` | Launches the GalaxeaVLA RoboDojo training recipe through the XPolicyLab argument convention. |
| `validate_train_entry.sh` | Runs a short dry-run train validation without writing checkpoints. |
| `eval.sh` | Starts a same-machine policy server and RoboDojo environment client. |
| `setup_eval_policy_server.sh` | Starts only the policy server. Use this for split-machine evaluation. |
| `setup_eval_env_client.sh` | Starts only the RoboDojo environment client and connects to a policy server. |
| `deploy.py` | XPolicyLab policy deployment wrapper. |
| `deploy.yml` | Default runtime configuration. |
| `model.py` | G05 model adapter used by the policy server. |

## Requirements

- A GalaxeaVLA/G0.5 source checkout compatible with the submitted checkpoint.
- A Python environment that can run that GalaxeaVLA checkout.
- RoboDojo LeRobot v3.0 data for training.
- For simulator evaluation, the RoboDojo evaluator workspace must be available
  next to XPolicyLab as expected by the official XPolicyLab scripts.

Common environment variables:

| Variable | Required | Description |
|---|---:|---|
| `G05_ROOT` | yes | Path to the GalaxeaVLA source checkout. |
| `G05_PYTHON` | optional | Python executable or venv Python for the G05 policy runtime. Defaults to `python3`. |
| `ROBODOJO_LEROBOT_V30_ROOT` | training | Path to the RoboDojo LeRobot v3.0 dataset root. |
| `G05_CKPT_PATH` | evaluation | Path to a G05 `.pt` checkpoint or run directory. |
| `G05_TRAIN_MODE` | optional | `fm_only` by default; `ar_fm` is also accepted if the GalaxeaVLA config exists. |
| `G05_GLOBAL_BATCH` | optional | Default `256`, matching the submitted FM-only training recipe. |
| `G05_BATCH_SIZE_PER_GPU` | optional | Default `8`. |
| `G05_TRAIN_SEED` | optional | Default `7`, matching the submitted FM-only training recipe. |
| `G05_MAX_STEPS` | optional | Overrides training max steps. Not set by default. |
| `G05_RESUME_CKPT` | optional | Resume checkpoint path forwarded to GalaxeaVLA Hydra config. |

## Installation

```bash
cd XPolicyLab/policy/G05
export G05_ROOT=/path/to/GalaxeaVLA
export G05_PYTHON=/path/to/python        # optional
bash install.sh
```

`install.sh` installs XPolicyLab itself in editable mode and attempts to install
`G05_ROOT` in editable mode if the checkout provides `pyproject.toml` or
`setup.py`.

## Training

`train.sh` follows the standard XPolicyLab policy signature:

```bash
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [hydra_overrides...]
```

For the submitted RoboDojo FM-only recipe, use joint control on ARX X5:

```bash
cd XPolicyLab/policy/G05
export G05_ROOT=/path/to/GalaxeaVLA
export G05_PYTHON=/path/to/python
export ROBODOJO_LEROBOT_V30_ROOT=/path/to/RoboDojo_lerobot_v30_video

bash train.sh RoboDojo cotrain arx_x5 joint 0 0,1,2,3,4,5,6,7
```

Defaults used by this adapter for reproducibility:

```text
G05_TRAIN_MODE=fm_only
G05_GLOBAL_BATCH=256
G05_BATCH_SIZE_PER_GPU=8
G05_TRAIN_SEED=7
```

`train.sh` computes gradient accumulation from the requested GPU count:

```text
grad_accumulation_steps = G05_GLOBAL_BATCH / (num_gpus * G05_BATCH_SIZE_PER_GPU)
```

Examples:

```text
16 GPUs × batch 8 × grad_accum 2  = global batch 256
 8 GPUs × batch 8 × grad_accum 4  = global batch 256
 2 GPUs × batch 8 × grad_accum 16 = global batch 256
```

Set `G05_MAX_STEPS` to control the training length, or pass additional Hydra
overrides after the six standard arguments.

## Train Entry Validation

Use `validate_train_entry.sh` to verify the training entry before launching a
long job. It runs GalaxeaVLA with `--dry-run --max_datasets 1`, computes the
same effective batch settings, runs one train step plus one eval step, and does
not write checkpoints.

```bash
cd XPolicyLab/policy/G05
export G05_ROOT=/path/to/GalaxeaVLA
export G05_PYTHON=/path/to/python
export ROBODOJO_LEROBOT_V30_ROOT=/path/to/RoboDojo_lerobot_v30_video
export G05_VALIDATE_GPUS=0,1,2,3,4,5,6,7
bash validate_train_entry.sh
```

A healthy run prints a line like:

```text
[G05 train] mode=fm_only ... train_seed=7 ... per_gpu_batch=8 ... global_batch=256
```

## Evaluation

Set `G05_CKPT_PATH` to a G05 run directory or `.pt` checkpoint. If the checkpoint
is in the standard submitted layout, the sidecars should live next to it:

```text
XPolicyLab/policy/G05/
├── .hydra/config.yaml
├── dataset_stats.json
├── action_tokenizer.pt
└── checkpoints/checkpoint
```

Same-machine evaluation:

```bash
cd XPolicyLab/policy/G05
export G05_ROOT=/path/to/GalaxeaVLA
export G05_PYTHON=/path/to/python
export G05_CKPT_PATH=$(pwd)/checkpoints/checkpoint

bash eval.sh RoboDojo stack_bowls g05 arx_x5 joint 0 0 0 \
  "$G05_PYTHON" <eval_env_conda_env>
```

Split policy-server / environment-client evaluation:

```bash
# Policy machine
bash setup_eval_policy_server.sh \
  RoboDojo stack_bowls g05 arx_x5 joint 0 \
  0 "$G05_PYTHON" 5000 0.0.0.0

# Environment machine
bash setup_eval_env_client.sh \
  RoboDojo stack_bowls g05 arx_x5 joint 0 \
  0 <eval_env_conda_env> "ckpt_name=g05,action_type=joint" \
  5000 <policy_server_ip>
```

## Action Source

The submitted checkpoint is FM-only. By default the adapter uses `action_source=auto`,
which resolves to FM when the loaded checkpoint has `continuous_action=true`.

You can explicitly set the action source with:

```bash
export ROBODOJO_G05_ACTION_SOURCE=fm
```

When FM is selected, the adapter forwards Hydra overrides equivalent to:

```text
model.model_arch.discrete_action=false
model.model_arch.continuous_action=true
model.model_arch.return_continuous_action=true
model.processor.discrete_action=false
```

`action_type=joint` and `env_cfg_type=arx_x5` are currently supported. The action
dimension is 14, and the policy predicts a 32-step action horizon; `deploy.yml`
executes `action_steps: 16` per replan by default.

## Submitted Checkpoint

The submitted FM-only RoboDojo ARX X5 joint checkpoint and sidecar assets are
hosted separately, for example on Hugging Face or another model hosting service.
After downloading/extracting them, set:

```bash
export G05_CKPT_PATH=/path/to/XPolicyLab/policy/G05/checkpoints/checkpoint
```

The checkpoint file may be named simply `checkpoint`; the training step is not
encoded in the public filename.
