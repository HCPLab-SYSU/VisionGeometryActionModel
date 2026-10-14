<div align="center">

# [ACM-MM 2026] Robotic Manipulation is Vision-to-Geometry Mapping

**Vision-Geometry backbones over language and video models for precise robotic manipulation.**

[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.2](https://img.shields.io/badge/PyTorch-2.2.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Zijian Song<sup>1</sup>, Qichang Li<sup>1</sup>, Jiawei Zhou<sup>1</sup>, Zhenlong Yuan<sup>4</sup>, Tianshui Chen<sup>3,5</sup>, Liang Lin<sup>1,2,3</sup>, Guangrun Wang<sup>1,2,3</sup>***

<sup>1</sup>Sun Yat-sen University · <sup>2</sup>Guangdong Key Laboratory of Big Data Analysis and Processing · <sup>3</sup>X-Era AI Lab · <sup>4</sup>LongCat, Meituan · <sup>5</sup>Guangdong University of Technology

[Project Page](https://hcplab-sysu.github.io/VisionGeometryActionModel/) · [Paper](https://arxiv.org/abs/2604.12908) · [Code](https://github.com/HCPLab-SYSU/VisionGeometryActionModel)

</div>



---

<p align="center">
  <img src="docs/assets/img/overview/figure2.png" alt="" width="600">
</p>

<p align="center">
  <em>Fig 1: The illustration of Vision Geometry Action Model.</em>
</p>


---




## Abstract


VGA treats robotic manipulation as a vision-to-geometry mapping problem: visual observations are converted into a native 3D representation before actions are decoded. This repository provides reproducible LIBERO and LIBERO-plus training and evaluation entry points based on a pretrained VGGT backbone.

The public repository's `master` branch contains two modes:

| Mode | Conda environment | Training entry point | Evaluation entry point |
| --- | --- | --- | --- |
| Standard LIBERO | `vga_libero` | `bash run_libero.sh` | `bash eval_libero.sh` |
| LIBERO-plus | `vga_liberoplus` | `bash run_liberoplus.sh` | `bash eval_liberoplus.sh` |


## Contents

- [Model overview](#model-overview)
- [Repository and environment workflow](#repository-and-environment-workflow)
- [Installation](#installation)
- [Datasets and assets](#datasets-and-assets)
- [Pretrained models](#pretrained-models)
- [Training](#training)
- [Evaluation](#evaluation)
- [Reproducing the reported results](#reproducing-the-reported-results)
- [Results and baseline comparison](#results-and-baseline-comparison)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## Model overview

VGA replaces a conventional language or video backbone with a pretrained VGGT 3D world model. Two camera views are processed by VGGT's alternating local and global attention, producing native spatial representations that preserve scene geometry. Learnable action queries and camera tokens connect these representations to the action head; proprioception is projected into the same multimodal sequence.


## Repository and environment workflow

After publication, the repository has a single public `master` branch containing both modes. Clone it once:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL> vggt_adapter
cd vggt_adapter

# Choose one environment for the mode you want to run:
conda activate vga_libero   # standard LIBERO
# conda activate vga_liberoplus  # LIBERO-plus
```

Do not change the evaluator source code when switching modes. The evaluator command and Conda environment select the mode, while environment variables select local installations and model locations. A typical setup is:

```bash
export PROJECT_ROOT="$(pwd)"
export LIBERO_ROOT="$PROJECT_ROOT/../LIBERO"          # standard LIBERO mode
export LIBEROPLUS_ROOT="$PROJECT_ROOT/../LIBERO-plus" # LIBERO-plus mode
export LIBERO_CONFIG_PATH="$PROJECT_ROOT/data/libero_config" # use liberoplus_config in vga_liberoplus
export VGGT_ROOT="$PROJECT_ROOT/../vggt"              # only needed for a source checkout
export VGGT_MODEL_PATH="facebook/VGGT-1B"             # or an existing local VGGT-1B directory
export PROMPT_ENCODER_PATH="Alibaba-NLP/gte-Qwen2-1.5B-instruct" # LIBERO-plus mode
```

`VGGT_MODEL_PATH` and `PROMPT_ENCODER_PATH` may be local directories. When they are Hugging Face IDs, the first run downloads them through the normal Hugging Face cache.

## Installation

The commands below mirror the environments used for the experiments. `our_envs.txt` is a package snapshot for reference; it is not intended to replace the steps below as a lock file.

### 1. System prerequisites

Use a Linux machine with an NVIDIA GPU, a working CUDA driver, Conda, and Git LFS. The reference environment uses Python 3.10, PyTorch 2.2.0, and CUDA 12.1 wheels.

```bash
conda create -n vga_libero python=3.10 -y
conda activate vga_libero

# Reference build used by the experiments (CUDA 12.1).
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
  --index-url https://download.pytorch.org/whl/cu121
# If your driver requires another CUDA build, use the matching command from pytorch.org instead.
pip uninstall -y numpy
pip install numpy==1.26.4
```

### 2. Install VGGT and this repository

The VGGT wrapper imports the official VGGT package. Either install a source checkout or install the package from the checkout you already have:

```bash
cd "$PROJECT_ROOT/.."
git clone https://github.com/facebookresearch/vggt.git
pip install -e ./vggt

cd "$PROJECT_ROOT"
pip install -e .
pip install packaging ninja
ninja --version
```

Flash-Attention is needed by the training path. The reference server uses 2.5.5:

```bash
pip install "flash-attn==2.5.5" --no-build-isolation
```

If compilation fails, download a wheel matching your Python, PyTorch, CUDA, and C++ ABI versions from the [Flash-Attention releases](https://github.com/Dao-AILab/flash-attention/releases), then install that wheel directly.

### 3. Install ordinary LIBERO (`vga_libero`)

Follow the official [LIBERO installation](https://github.com/Lifelong-Robot-Learning/LIBERO) and keep the checkout outside this repository:

```bash
cd "$PROJECT_ROOT/.."
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
export LIBERO_ROOT="$PWD/LIBERO"

# The upstream README lists a larger, pinned legacy environment. Keep the
# project-wide PyTorch/Transformers versions above and install the simulator
# dependencies used by this repository:
pip install -e "$LIBERO_ROOT"

cd "$PROJECT_ROOT"
pip install -r experiments/robot/libero/libero_requirements.txt
```

Download the ordinary LIBERO assets and initial states using the upstream helper:

```bash
cd "$LIBERO_ROOT"
python benchmark_scripts/download_libero_datasets.py --use-huggingface
cd "$PROJECT_ROOT"
```

The upstream `requirements.txt` and setup instructions remain the reference for a standalone LIBERO installation; do not install its legacy Torch 1.11 pin on top of the `vga_libero` environment.

If the simulator reports an EGL error such as `eglQueryString`, install the system libraries used by the official simulator setup:

```bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev libglew-dev
```

### 4. Create the LIBERO-plus environment (`vga_liberoplus`)

Start from the working ordinary-LIBERO environment. LIBERO-plus replaces the `libero` Python package, so keeping a cloned environment avoids dependency conflicts:

```bash
conda create -n vga_liberoplus --clone vga_libero
conda activate vga_liberoplus

cd "$PROJECT_ROOT/.."
git clone https://github.com/sylvestf/LIBERO-plus.git
export LIBEROPLUS_ROOT="$PWD/LIBERO-plus"

cd "$LIBEROPLUS_ROOT"
pip uninstall -y libero
pip install -e .
pip install -r extra_requirements.txt
```

On a clean Ubuntu machine, LIBERO-plus may also require the packages listed in its official guide:

```bash
sudo apt-get install -y libexpat1 libfontconfig1-dev libpython3-stdlib libmagickwand-dev
```

### 5. Keep a separate simulator configuration for each mode

Both upstream packages support `LIBERO_CONFIG_PATH` (a directory containing `config.yaml`). Create two configuration directories once, then select the appropriate directory along with the environment. This avoids editing the shared `~/.libero/config.yaml` each time you switch modes.

```bash
python - <<'PYCONFIG'
import os
from pathlib import Path

project = Path(os.environ["PROJECT_ROOT"]).resolve()
for mode, variable in (("libero", "LIBERO_ROOT"), ("liberoplus", "LIBEROPLUS_ROOT")):
    root = Path(os.environ[variable]).resolve() / "libero/libero"
    directory = project / "data" / f"{mode}_config"
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "benchmark_root": root,
        "assets": root / "assets",
        "bddl_files": root / "bddl_files",
        "init_states": root / "init_files",
        "datasets": root.parent / "datasets",
    }
    (directory / "config.yaml").write_text(
        "".join(f"{key}: {str(value)!r}\n" for key, value in paths.items())
    )
PYCONFIG
```

For standard LIBERO use `export LIBERO_CONFIG_PATH="$PROJECT_ROOT/data/libero_config"`; for LIBERO-plus use `export LIBERO_CONFIG_PATH="$PROJECT_ROOT/data/liberoplus_config"`. Always set it **before** importing `libero` or starting an evaluator. Setting `LIBERO_ROOT`/`LIBEROPLUS_ROOT` chooses the Python source checkout; setting `LIBERO_CONFIG_PATH` chooses the assets, BDDL, and initial-state paths. Both must refer to the same mode.

### 6. Verify the installation

Run these checks in the environment you intend to use:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
import prismatic
print("prismatic import: ok")
PY
```

For standard LIBERO, verify `import libero` after activating `vga_libero`. For LIBERO-plus, verify it after activating `vga_liberoplus` and confirm that `LIBEROPLUS_ROOT/libero/libero/benchmark/task_classification.json` exists.

## Datasets and assets

### RLDS training datasets

The training code uses the RLDS conversion published by OpenVLA. Download the four no-noop datasets from [modified_libero_rlds](https://huggingface.co/datasets/openvla/modified_libero_rlds):

```bash
mkdir -p data/libero
git lfs install
# Either clone with Git LFS or download the dataset files from the page above.
git clone https://huggingface.co/datasets/openvla/modified_libero_rlds data/libero
```

The directory passed to `--data_root_dir` must contain the dataset directories directly:

```text
data/libero/
├── libero_spatial_no_noops/1.0.0/
├── libero_object_no_noops/1.0.0/
├── libero_goal_no_noops/1.0.0/
└── libero_10_no_noops/1.0.0/
```

If the download creates an extra `modified_libero_rlds/` directory, either move its contents into `data/libero/` or pass the parent directory that contains the four dataset directories. The `--task_suite` value in training (`spatial`, `object`, `goal`, or `10`) selects the corresponding RLDS dataset automatically.

### LIBERO-plus assets

LIBERO-plus uses the ordinary LIBERO task suites with additional environment perturbations. Download its assets from the [LIBERO-plus dataset page](https://huggingface.co/datasets/Sylvest/LIBERO-plus/tree/main) and unpack them under:

```text
$LIBEROPLUS_ROOT/libero/libero/assets/
├── articulated_objects/
├── new_objects/
├── scenes/
├── stable_hope_objects/
├── stable_scanned_objects/
├── textures/
└── ...
```

The training RLDS files remain the ordinary LIBERO files; LIBERO-plus changes the evaluation environment and task classification.

## Pretrained models

The following models are not committed to Git because of their size.

1. **Prismatic VLM base model**

   Download [Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b](https://huggingface.co/Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b) into `pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b`:

   ```bash
   huggingface-cli download \
     Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b \
     --local-dir pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b
   ```

   The checked-in `pretrained_models/configs/` files provide the project-specific OpenVLA configuration and processor metadata used by both training scripts. Model weights stay in the downloaded base-model directory; no model weights need to be committed to this repository.

   The Prismatic loader also resolves the Qwen2.5-0.5B tokenizer/config from [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B). Make sure the machine can reach Hugging Face, or pre-cache that repository before starting training.

2. **VGGT-1B**

   Set `VGGT_MODEL_PATH` to a local checkout of [facebook/VGGT-1B](https://huggingface.co/facebook/VGGT-1B), or leave it as the Hugging Face ID so that Transformers downloads it on first use:

   ```bash
   export VGGT_MODEL_PATH="facebook/VGGT-1B"
   # Example for a local copy:
   # export VGGT_MODEL_PATH=/models/VGGT-1B
   ```

3. **GTE prompt encoder**

   LIBERO-plus training and evaluation use [Alibaba-NLP/gte-Qwen2-1.5B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct). Set `PROMPT_ENCODER_PATH` to its local directory or its Hugging Face ID:

   ```bash
   export PROMPT_ENCODER_PATH="Alibaba-NLP/gte-Qwen2-1.5B-instruct"
   ```

   The prompt cache is generated per suite at `data/libero_<suite>_prompt_cache.pt`. Keep the encoder path and the checkpoint's recorded GTE dimensions consistent when evaluating.

## Training

The checked-in `run_libero.sh` and `run_liberoplus.sh` files are complete experiment templates. The suffix identifies the simulator mode: `_libero` is for ordinary LIBERO and `_liberoplus` is for LIBERO-plus. The LIBERO recipe does not enable visual-token perturbation or pass any perturbation probability. The plus recipe explicitly enables the perturbation settings used by the LIBERO-plus experiments. Both use four GPUs by default, so edit `CUDA_VISIBLE_DEVICES`, batch size, save frequency, and the suite before launching a long run.

The shared training settings are summarized below; the shell scripts remain the authoritative source for the full command line.

| Mode | GPUs | Batch / GPU | Gradient accumulation | Effective batch | Initial LR | LoRA rank | Save frequency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LIBERO | 4 (`4,5,6,7`) | 16 | 1 | 64 | `2e-4` | 64 | 2k |
| LIBERO-plus | 4 (`0,1,2,3`) | 16 | 1 | 64 | `2e-4` | 64 | 5k |

Both recipes use two input images, proprioception, LoRA fine-tuning, merged LoRA checkpoints, and the Pro action head. The plus recipe additionally enables GTE and visual-token perturbation. See [`run_libero.sh`](run_libero.sh) and [`run_liberoplus.sh`](run_liberoplus.sh) before changing any experiment setting.

The paper reports these experiments on NVIDIA A100-SXM4-80GB GPUs. The scripts expose `CUDA_VISIBLE_DEVICES`, so the same recipes can be adapted to another multi-GPU machine after checking memory and CUDA compatibility.

### Standard LIBERO

```bash
conda activate vga_libero
export LIBERO_ROOT="$PROJECT_ROOT/../LIBERO"
export LIBERO_CONFIG_PATH="$PROJECT_ROOT/data/libero_config"
export VGGT_MODEL_PATH="facebook/VGGT-1B"
bash run_libero.sh
```

The standard recipe sets the suite in `--task_suite`. Change it to `spatial`, `object`, `goal`, or `10`. The training script generates the run directory name from the suite, mode, and training settings; no experiment-version selector is required. The output layout is:

```text
outputs/<suite>/<run>/training_args.json
outputs/<suite>/<run>--<step>_chkpt/
```

For example, the default LIBERO recipe writes step checkpoints under `outputs/goal/goal_libero+b16+lr-0.0002+lora-r64+dropout-0.0+image_aug--<step>_chkpt/`. The `b` value is the per-GPU batch size multiplied by gradient accumulation; multiply it by the number of GPUs for the total batch size. LIBERO-plus run names include `liberoplus`, so the two modes use separate output directories. When resuming with `--resume True --resum_vla_path <checkpoint> --resume_step <step>`, the run name is recovered from the checkpoint path, including older experiment names. Running the same recipe again uses the same output directory; archive an earlier run before starting over if you need to retain it.

The LIBERO entry keeps its BOS-offset multimodal token layout (`--model_mode libero`). The LIBERO-plus entry switches to the GTE prompt path and its no-BOS layout (`--model_mode liberoplus`). The selected mode is saved in `training_args.json` and restored by the evaluator, so evaluation does not depend on the current shell command alone. `--use_pro_version` selects the action-head behavior and is unrelated to experiment naming.

#### BOS-token compatibility note

The two model modes intentionally keep different multimodal token layouts. In the original LIBERO experiments, the BOS token was handled incorrectly when visual tokens were inserted; the public LIBERO entry preserves that behavior because the corresponding checkpoints were trained with it. LIBERO-plus fixes the insertion order and uses the corrected BOS handling, which we believe is the more principled implementation. We did not rerun the standard LIBERO experiments with the corrected layout, so the reported LIBERO numbers and any future LIBERO weights remain tied to the historical behavior.

### LIBERO-plus recipe

```bash
conda activate vga_liberoplus
export LIBEROPLUS_ROOT="$PROJECT_ROOT/../LIBERO-plus"
export LIBERO_CONFIG_PATH="$PROJECT_ROOT/data/liberoplus_config"
export VGGT_MODEL_PATH="facebook/VGGT-1B"
export PROMPT_ENCODER_PATH="Alibaba-NLP/gte-Qwen2-1.5B-instruct"
bash run_liberoplus.sh
```

The `run_liberoplus.sh` template enables the following options used by the recorded LIBERO-plus runs:

```text
--use_gte_encoder True
--prompt_encoder_path $PROMPT_ENCODER_PATH
--use_visual_token_perturbation True
--visual_token_pert_prob 0.8
--visual_token_pert_full_shuffle_prob 0.1
--num_images_in_input 2
--use_proprio True
```

`--use_lora True` trains a LoRA adapter and `--merge_lora_during_training True` also writes a merged model into the checkpoint. Set `--use_lora False --use_fz True` for the full/frozen variant. Each run stores `training_args.json`, including the selected mode (`libero` or `liberoplus`) and any GTE dimensions needed by the evaluator.

For a smaller GPU, reduce `--batch_size`, increase `--grad_accumulation_steps` to preserve the effective batch size, and lower `--lora_rank` only if memory is still insufficient. The VGGT path is memory-heavy, so start with one GPU and a short `--max_steps` to validate the configuration before launching distributed training.

## Evaluation

Evaluation scripts expect the original training output layout: `outputs/<suite>/<run>/training_args.json` plus the sibling `outputs/<suite>/<run>--<step>_chkpt/` directory containing `full_vla_model.bin`, processor files, the action head, and the saved projectors. Keep the run metadata beside the step checkpoint when copying a checkpoint to another machine. Results are written under `experiments/logs/`; rollout videos are written under `rollouts/`.

Both evaluators read the saved model mode and `use_gte_encoder` from `training_args.json`. A non-GTE checkpoint can be evaluated in either benchmark without a prompt encoder. For a GTE checkpoint, supply `--prompt_encoder_path "$PROMPT_ENCODER_PATH"` in either evaluator. The optional Adapt3R evaluator does not initialize GTE; use `run_libero_eval.py` for standard LIBERO evaluation of a GTE checkpoint. Visual-token perturbation applies during training and is disabled by the model in evaluation mode.

### Ordinary LIBERO

The supported entry point is `eval_libero.sh`; it uses the same standard LIBERO evaluator and command as the released ordinary-LIBERO recipe. Edit its checkpoint path when needed, then run all tasks in a suite with the default 50 trials per task:

```bash
conda activate vga_libero
export LIBERO_ROOT="$PROJECT_ROOT/../LIBERO"
export LIBERO_CONFIG_PATH="$PROJECT_ROOT/data/libero_config"
bash eval_libero.sh
```

To run the equivalent command directly:

```bash
$CONDA_PREFIX/bin/python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint outputs/object/object_lora_011--36000_chkpt \
  --task_suite_name libero_object \
  --use_pro_version True \
  --use_proprio True \
  --num_images_in_input 2 \
  --num_trials_per_task 50
```

`eval_libero.sh` is the standard LIBERO evaluation entry and matches the released ordinary-LIBERO evaluator command.

### LIBERO-plus

Use `eval_liberoplus.sh` in `vga_liberoplus`. Edit its checkpoint path, suite, and category before launching. Set `LIBEROPLUS_ROOT` so the script can locate `task_classification.json`; the path can also be supplied explicitly with `--task_classification_path`.

```bash
conda activate vga_liberoplus
export LIBEROPLUS_ROOT="$PROJECT_ROOT/../LIBERO-plus"
export LIBERO_CONFIG_PATH="$PROJECT_ROOT/data/liberoplus_config"
export PROMPT_ENCODER_PATH="Alibaba-NLP/gte-Qwen2-1.5B-instruct"
export VGGT_MODEL_PATH="facebook/VGGT-1B"
bash eval_liberoplus.sh
```

To run the equivalent command directly:

```bash
$CONDA_PREFIX/bin/python experiments/robot/libero/run_liberoplus_eval.py \
  --pretrained_checkpoint outputs/goal/goal_lora_320--30000_chkpt \
  --task_suite_name libero_goal \
  --task_category "Camera Viewpoints" \
  --prompt_encoder_path "$PROMPT_ENCODER_PATH" \
  --use_pro_version True \
  --use_proprio True \
  --num_images_in_input 2 \
  --num_trials_per_task 1
```

Supported `--task_category` values are:

```text
Camera Viewpoints
Robot Initial States
Language Instructions
Light Conditions
Background Textures
Sensor Noise
Objects Layout
```

The recorded runs use one trial per perturbation task. To evaluate a complete suite without category filtering, omit `--task_category`; this is a longer run and uses the number of trials set by `--num_trials_per_task`.

The evaluation examples retain historical checkpoint names from the recorded experiments. For a newly trained model, replace `--pretrained_checkpoint` in the corresponding evaluation script with the generated step-checkpoint directory shown during training, and set `--task_suite_name` to match its suite. Older checkpoints can still be evaluated without renaming their directories.

## Reproducing the reported results

The commands in `run_libero.sh` and `run_liberoplus.sh` are full recipes. They use four GPUs, LoRA rank 64, per-GPU batch size 16, gradient accumulation 1 (effective batch size 64), and an initial learning rate of `2e-4`. The LIBERO-plus recipe also enables the GTE prompt encoder and visual-token perturbation. See the scripts for the complete argument list, model paths, save frequency, and perturbation settings. The final schedule is also kept in [`result_summary.md`](result_summary.md).

### Standard LIBERO schedule

The standard results use the following schedules. A two-stage row means that the second stage resumes from the checkpoint at the end of stage one; `max_steps` is the global step after resuming.

| Suite | Stage 1 | Stage 2 | Final checkpoint | Success rate |
| --- | --- | --- | ---: | ---: |
| Spatial | `2e-4` to 6k | `2e-5` from 6k to 15k | 15k | **99.0%** |
| Object | `2e-4` to 43k | — | 43k | **99.6%** |
| Goal | `2e-4` to 34k | `5e-5` from 34k to 106k | 106k | **98.6%** |
| Long (`10`) | `2e-4` to 69k | `2e-5` from 69k to 126k | 126k | **95.0%** |

For a one-stage run, edit `--task_suite`, `--learning_rate`, and `--max_steps` in `run_libero.sh`. For a two-stage run, add `--resume True`, `--resum_vla_path <stage-1-checkpoint>`, and `--resume_step <stage-1-step>` to the second command, then set its learning rate and final `--max_steps`. For example, the Spatial continuation uses `--resum_vla_path outputs/spatial/<run>--6000_chkpt`, `--resume_step 6000`, `--learning_rate 2e-5`, and `--max_steps 15000`. The same pattern applies to Goal (`34k → 106k`) and Long (`69k → 126k`). The scheduler in the script has a milestone at 200k, so these staged schedules are implemented by starting the continuation with the lower learning rate shown above.

### LIBERO-plus schedule

In the LIBERO-plus table below, the suffix (`50k`, `30k`, or `160k`) is the step of the checkpoint used for that row. These are the final release-reproduction results from `result_summary.md`, rather than the earlier exploratory log selection.

| Suite / checkpoint | CV | RIS | LI | LC | BT | SN | OL | Avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Spatial · **50k** | 94.4 | 56.9 | 86.9 | 93.5 | 84.5 | 93.2 | 93.8 | **86.2** |
| Object · **30k** | 97.5 | 32.9 | 95.8 | 97.0 | 84.7 | 98.1 | 68.2 | **82.0** |
| Goal · **30k** | 88.0 | 27.9 | 41.0 | 86.4 | 60.5 | 91.8 | 63.1 | **65.5** |
| Long · **160k** | 65.4 | 42.7 | 96.1 | 86.5 | 45.0 | 76.2 | 67.6 | **68.5** |
| Average | 86.3 | 40.1 | 80.0 | 90.9 | 68.7 | 89.8 | 73.2 | **75.6** |

`CV`, `RIS`, `LI`, `LC`, `BT`, `SN`, and `OL` denote Camera Viewpoints, Robot Initial States, Language Instructions, Light Conditions, Background Textures, Sensor Noise, and Objects Layout. To reproduce a row, edit `--task_suite` and `--max_steps` in `run_liberoplus.sh`, then evaluate with the matching category in `eval_liberoplus.sh`. Keep `--learning_rate 2e-4`, batch size 16, four GPUs, and the other options in the checked-in script unless you are intentionally changing the experiment.

## Results and baseline comparison

The tables below report success rates. LIBERO is copied from the paper's full baseline comparison. LIBERO-plus uses the final release-reproduction values above; the paper baseline rows are included only for the three perturbation dimensions reported there (light, language, and camera), so unreported cells are shown as `—`.

### LIBERO

| Family | Method | Spatial | Object | Goal | Long | Avg |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| VLA | Octo | 78.9 | 85.7 | 84.6 | 51.1 | 75.1 |
| VLA | OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| VLA | GR00T-N1 | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 |
| VLA | UniVLA | 96.5 | 96.8 | 95.6 | 92.0 | 95.2 |
| VLA | π0 | 90.0 | 86.0 | 95.0 | 73.0 | 86.0 |
| VLA | π0.5 | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| VLA | GR00T-N1.6 | 97.7 | 98.5 | 97.5 | 94.4 | 97.0 |
| VLA | OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| VLA | VLA-Thinker | 98.7 | 99.0 | 95.2 | 96.9 | 97.5 |
| 3D-VLA | SpatialVLA | 88.2 | 89.9 | 78.6 | 55.5 | 78.1 |
| 3D-VLA | GeoAwareVLA | 95.0 | 100.0 | 99.0 | 93.0 | 96.8 |
| 3D-VLA | GeoVLA | 98.4 | 99.0 | 96.6 | 96.6 | 97.7 |
| WAM | UniMimic | 89.0 | 91.0 | 85.0 | 59.0 | 81.0 |
| WAM | WorldVLA | 87.6 | 96.2 | 83.4 | 60.0 | 81.8 |
| WAM | UVA | — | — | — | 93.0 | 93.0 |
| WAM | mimic-video | 94.2 | 96.8 | 90.6 | — | 93.9 |
| WAM | Motus | 96.8 | 99.8 | 96.6 | 97.6 | 97.7 |
| Ours | **VGA** | **99.0** | **99.6** | **98.6** | **95.0** | **98.1** |

### LIBERO-plus

| Method / checkpoint | CV | RIS | LI | LC | BT | SN | OL | Avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **VGA · Spatial 50k** | **94.4** | **56.9** | **86.9** | **93.5** | **84.5** | **93.2** | **93.8** | **86.2** |
| **VGA · Object 30k** | **97.5** | **32.9** | **95.8** | **97.0** | **84.7** | **98.1** | **68.2** | **82.0** |
| **VGA · Goal 30k** | **88.0** | **27.9** | **41.0** | **86.4** | **60.5** | **91.8** | **63.1** | **65.5** |
| **VGA · Long 160k** | **65.4** | **42.7** | **96.1** | **86.5** | **45.0** | **76.2** | **67.6** | **68.5** |
| OpenVLA-OFT (paper) | 59.7 | — | 81.5 | 85.8 | — | — | — | — |
| π0 (paper) | 15.8 | — | 61.0 | 79.6 | — | — | — | — |
| WorldVLA (paper) | 0.3 | — | 44.2 | 29.4 | — | — | — | — |
| UniVLA (paper) | 4.3 | — | 71.8 | 59.1 | — | — | — | — |
| RIPT-VLA (paper) | 58.3 | — | 80.1 | 87.9 | — | — | — | — |

The LIBERO-plus baseline rows follow the paper's column order of camera, language, and light; they are not recomputed as seven-category averages. The release-reproduction rows are the values to use when checking this repository.

## Troubleshooting

- **EGL or renderer errors:** install the Mesa/EGL packages listed in the LIBERO installation section and run the evaluator on a machine with an NVIDIA driver that supports off-screen rendering.
- **Wrong LIBERO package is imported:** print `import libero; print(libero.__file__)` after activating the environment. `vga_libero` must resolve the ordinary LIBERO checkout; `vga_liberoplus` must resolve `LIBERO-plus`.
- **LIBERO-plus cannot find task categories:** set `LIBEROPLUS_ROOT` or pass `--task_classification_path` and check that `libero/libero/benchmark/task_classification.json` exists.
- **GTE dimension mismatch:** use the same GTE model family during training and evaluation. The evaluator compares the loaded encoder's `hidden_size` with the values saved in `training_args.json`.
- **Flash-Attention build errors:** install a matching prebuilt wheel and verify `python -c "import flash_attn; print(flash_attn.__version__)"`.
- **B200/Pro6000-class GPUs:** the reference CUDA 12.1 wheel may be unsuitable. Use a PyTorch/CUDA build supported by the driver (the server setup used PyTorch 2.7.1 + cu128 and Flash-Attention 2.8.3) and then rerun the import checks.
- **Out-of-memory during training:** first lower per-device `--batch_size`, then compensate with `--grad_accumulation_steps`; keep the effective batch size and number of GPUs in mind when comparing results.

## Acknowledgements

We thank the authors of [VLA-Adapter](https://github.com/OpenHelix-Team/VLA-Adapter) for releasing a useful open-source repository that helped with the surrounding VLA training and evaluation infrastructure.

## Citation

If you find VGA useful, please cite:

```bibtex
@article{song2026robotic,
  title   = {Robotic manipulation is vision-to-geometry mapping (f(v) \rightarrow G): Vision-geometry backbones over language and video models},
  author  = {Song, Zijian and Li, Qichang and Zhou, Jiawei and Yuan, Zhenlong and Chen, Tianshui and Lin, Liang and Wang, Guangrun},
  journal = {arXiv preprint arXiv:2604.12908},
  year    = {2026}
}
```
