"""
finetune.py

Fine-tunes Qwen2.5-0.5B via LoRA.
"""
import os
import sys
libero_root = os.environ.get("LIBERO_ROOT")
if libero_root and libero_root not in sys.path:
    sys.path.insert(0, libero_root)
import time
import math
from collections import deque
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type
import torch.nn.functional as F
import draccus
import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
from accelerate import PartialState
from huggingface_hub import HfApi, snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModel, AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
import wandb
import json

from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map
)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction, get_valid_input_ids, last_token_pool
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import PromptProjector, ProprioProjector
from prismatic.training.train_utils import (
    compute_actions_l1_loss,
    compute_token_accuracy,
    get_current_action_mask,
    get_next_actions_mask
)
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
    NUM_TOKENS,
)
from prismatic.vla.datasets import RLDSDataset, RLDSBatchTransform
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from prismatic.models import load, load_vla

from prismatic.vggt_backbone.vggt_backbone_warpper import VGGTWarpperBackbone

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class FinetuneConfig:
    # fmt: off
    config_file_path: str = "openvla/openvla-7b"     # Path to necessary config files of LA-Adapter
    vlm_path: str = "openvla/openvla-7b"             # Path to OpenVLA model (on HuggingFace Hub or stored locally)
    use_minivlm: bool = False                        # 
    resum_vla_path: str = "openvla/openvla-7b"       # Path to OpenVLA model (on HuggingFace Hub or stored locally)

    task_suite: str = "none"  # 10, goal, object, spatial
    model_mode: str = "libero"  # Select the final LIBERO or LIBERO-plus model layout.

    # GTE prompt encoder
    use_gte_encoder: bool = False
    prompt_encoder_path: str = os.environ.get("PROMPT_ENCODER_PATH", "")
    prompt_cache_path: Path = Path("data/libero_prompt_cache.pt")
    prompt_cache_source_path: Optional[Path] = None
    force_rebuild_prompt_cache: bool = False
    prompt_pad_length_multiplier: float = 4.0

    # Dataset
    data_root_dir: Path = Path("datasets/rlds")      # Directory containing RLDS datasets
    dataset_name: str = "aloha_scoop_x_into_bowl"    # Name of fine-tuning dataset (e.g., `aloha_scoop_x_into_bowl`)
    run_root_dir: Path = Path("runs")                # Path to directory to store logs & checkpoints
    shuffle_buffer_size: int = 100_000               # Dataloader shuffle buffer size (can reduce if OOM errors occur)

    # Algorithm and architecture
    use_l1_regression: bool = True                   # If True, trains continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, trains continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for training 
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 1                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = False                        # If True, includes robot proprioceptive state in input
    phase1_path: str = "None"

    # Training configuration
    batch_size: int = 8                              # Batch size per device (total batch size = batch_size * num GPUs)
    learning_rate: float = 5e-4                      # Learning rate
    lr_warmup_steps: int = 0.1                       # Number of steps to warm up learning rate (from 10% to 100%)
    num_steps_before_decay: int = 100000             # Number of steps before LR decays by 10x
    grad_accumulation_steps: int = 1                 # Number of gradient accumulation steps
    max_steps: int = 200000                          # Max number of training steps
    use_val_set: bool = False                        # If True, uses validation set and log validation metrics
    val_freq: int = 10_000                           # (When `use_val_set==True`) Validation set logging frequency in steps
    val_time_limit: int = 180                        # (When `use_val_set==True`) Time limit for computing validation metrics
    save_freq: int = 10_000                          # Checkpoint saving frequency in steps
    save_latest_checkpoint_only: bool = False        # If True, saves only 1 checkpoint, overwriting latest checkpoint
                                                     #   (If False, saves all checkpoints)
    resume: bool = False                             # If True, resumes from checkpoint
    resume_step: Optional[int] = None                # (When `resume==True`) Step number that we are resuming from
    image_aug: bool = True                           # If True, trains with image augmentations (HIGHLY RECOMMENDED)
    diffusion_sample_freq: int = 50                  # (When `use_diffusion==True`) Frequency for sampling in steps

    # LoRA
    use_lora: bool = False                           # If True, uses LoRA fine-tuning
    lora_rank: int = 32                              # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                        # Dropout applied to LoRA weights
    merge_lora_during_training: bool = False         # If True, merges LoRA weights and saves result during training
                                                     #   Note: Merging can be very slow on some machines. If so, set to
                                                     #         False and merge final checkpoint offline!

    # Visual-token perturbation is enabled explicitly by the LIBERO-plus recipe.
    use_visual_token_perturbation: bool = False            # If True, applies visual-token perturbation during training
    visual_token_pert_prob: float = 0.8               # Per-sample probability of applying perturbation, 0.0~1.0
    visual_token_pert_full_shuffle_prob: float = 0.1  # Within perturbed samples, prob. of full intra-view shuffle, 0.0~1.0
    visual_token_pert_num_blocks_min: int = 3         # Lower bound for #blocks to replace per perturbed sample
    visual_token_pert_num_blocks_max: int = 6         # Upper bound for #blocks to replace per perturbed sample
    visual_token_pert_block_size_min: int = 3         # Lower bound for block edge length, 1~16
    visual_token_pert_block_size_max: int = 7         # Upper bound for block edge length, 1~16

    # Full Finetune
    use_fz: bool = False                             # use_fz = not use_lora

    # Logging
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project
    wandb_log_freq: int = 10                         # WandB logging frequency in steps

    # Action-head behavior (this is not an experiment or run version).
    use_pro_version: bool = True                             # Use the VLA-Adapter Pro action head
    phase: str = "Training"
    # fmt: on



def remove_ddp_in_checkpoint(state_dict) -> dict:
    """
    Removes the 'module.' prefix from parameter names in a PyTorch model state dictionary that was saved using
    DistributedDataParallel (DDP).

    When a model is trained using PyTorch's DistributedDataParallel, the saved state dictionary contains parameters
    prefixed with 'module.'. This function removes these prefixes to make the state dictionary compatible when
    loading into models that are not yet wrapped in DDP.

    Args:
        state_dict (dict): PyTorch model state dictionary.

    Returns:
        dict: A new state dictionary with the same contents but with 'module.' prefixes removed from parameter names.
              Parameters without the 'module.' prefix remain unchanged.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k[:7] == "module.":
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict



def get_run_id(cfg) -> str:
    """
    Generates or retrieves an identifier string for an experiment run.

    Args:
        cfg (FinetuneConfig): Training configuration.

    Returns:
        str: Experiment run ID.
    """
    if cfg.resume:
        run_id = Path(cfg.resum_vla_path).name
        if run_id.endswith("_chkpt") and "--" in run_id:
            run_id = run_id.rsplit("--", 1)[0]
    else:
        run_id = (
            f"{cfg.task_suite}_{cfg.model_mode}"
            f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
            f"+lr-{cfg.learning_rate}"
        )
        if cfg.use_fz:
            run_id += f"+frozen+dropout-{cfg.lora_dropout}"
        if cfg.use_lora:
            run_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
        if cfg.image_aug:
            run_id += "+image_aug"
    return run_id



def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    """
    Loads a checkpoint for a given module.

    Args:
        module_name (str): Name of model component to load checkpoint for.
        path (str): Path to checkpoint directory.
        step (int): Gradient step number of saved checkpoint.
        device (str): String specifying how to remap storage locations (default = "cpu").

    Returns:
        dict: PyTorch model state dictionary.
    """
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)



def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False) -> DDP:
    """
    Wrap a module with DistributedDataParallel.

    Args:
        module (nn.Module): PyTorch module.
        device_id (str): Device ID.
        find_unused (bool): Whether to detect parameters without gradients in distributed training.

    Returns:
        DistributedDataParallel: PyTorch module wrapped with DDP.
    """
    return DDP(module, device_ids=[device_id], find_unused_parameters=find_unused, gradient_as_bucket_view=True)



def count_parameters(module: nn.Module, name: str) -> None:
    """
    Counts and prints the number of trainable parameters in a module.

    Args:
        module (nn.Module): PyTorch module.
        module_name (str): Name of model component.

    Returns:
        None.
    """
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    print(f"# trainable params in {name}: {num_params}")



def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: FinetuneConfig,
    device_id: int,
    module_args: dict,
    to_bf16: bool = False,
    find_unused_params: bool = False,
) -> DDP:
    """
    Initializes a module, optionally loads checkpoint, moves to device, and wraps with DDP.

    Args:
        module_class (Type[nn.Module]): Class of PyTorch module to initialize.
        module_name (str): Name of model component to load checkpoint for.
        cfg (FinetuneConfig): Training configuration.
        device_id (str): Device ID.
        module_args (dict): Args for initializing the module.
        to_bf16 (bool): Whether to convert to torch.bfloat16 data type.
        find_unused_params (bool): Whether to detect parameters without gradients in distributed training.

    Returns:
        DistributedDataParallel: PyTorch module wrapped with DDP.
    """
    module = module_class(**module_args)
    count_parameters(module, module_name)

    if cfg.resume:
        state_dict = load_checkpoint(module_name, cfg.resum_vla_path, cfg.resume_step)
        module.load_state_dict(state_dict)
        print('loaded!!!!!!!!!')

    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)

    return wrap_ddp(module, device_id, find_unused_params)



def run_forward_pass(
    vla,
    action_head,
    proprio_projector,
    prompt_projector,
    batch,
    action_tokenizer,
    device_id,
    use_l1_regression,
    use_proprio,
    use_film,
    num_patches,
    compute_diffusion_l1=False,
    use_pro_version=True,
    cfg=None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute model forward pass and metrics for both training and validation.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        batch (dict): Input batch.
        action_tokenizer (ActionTokenizer): Action tokenizer.
        device_id (str): Device ID.
        use_l1_regression (bool): Whether to use L1 regression.
        use_diffusion (bool): Whether to use diffusion.
        use_proprio (bool): Whether to use proprioceptive state as input.
        use_film (bool): Whether to use FiLM for better language following.
        num_patches (int): Number of vision patches.
        compute_diffusion_l1 (bool): Whether to sample actions and compute L1 loss for diffusion (do this once every
                                    diffusion_sample_freq steps during training; do it every batch for validation)
        num_diffusion_steps (int): Number of diffusion steps (only used for diffusion).

    Returns:
        tuple: (loss, metrics_dict)
            loss: The loss tensor with gradient for backpropagation.
            metrics_dict: Dictionary of computed metrics (detached values for logging).
    """
    metrics = {}

    # Get ground-truth action labels
    ground_truth_actions = batch["actions"].to(device_id).to(torch.bfloat16)
    noise, noisy_actions, diffusion_timestep_embeddings = None, None, None

    # VLA forward pass
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = vla(
            input_ids=batch["input_ids"].to(device_id),
            attention_mask=batch["attention_mask"].to(device_id),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
            labels=batch["labels"],
            output_hidden_states=True,
            proprio=batch["proprio"] if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
            prompt_projector=prompt_projector,
            noisy_actions=None,
            noisy_action_projector=None,
            diffusion_timestep_embeddings=None,
            use_film=use_film,
            )

    # Get action masks needed for logging
    # ground_truth_token_ids = batch["labels"][:,1:].to(device_id)
    # current_action_mask = get_current_action_mask(ground_truth_token_ids)
    # next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

    # Compute metrics for discrete action representation (next-token prediction)
    if not (use_l1_regression):
        assert False
        loss = output.loss
        predicted_token_ids = output.logits[:, num_patches:-1].argmax(dim=2)

        curr_action_accuracy = compute_token_accuracy(
            predicted_token_ids, 
            ground_truth_token_ids, 
            mask=current_action_mask
            )
        curr_action_l1_loss = compute_actions_l1_loss(
            action_tokenizer, 
            predicted_token_ids, 
            ground_truth_token_ids, 
            mask=current_action_mask
            )
        next_actions_accuracy = compute_token_accuracy(
            predicted_token_ids, 
            ground_truth_token_ids, 
            mask=next_actions_mask
            )
        next_actions_l1_loss = compute_actions_l1_loss(
            action_tokenizer, 
            predicted_token_ids, 
            ground_truth_token_ids, 
            mask=next_actions_mask
            )
        
        metrics.update(
            {
                "loss_value": loss.item(),  # Detached value for logging
                "curr_action_accuracy": curr_action_accuracy.item(),
                "curr_action_l1_loss": curr_action_l1_loss.item(),
                "next_actions_accuracy": next_actions_accuracy.item(),
                "next_actions_l1_loss": next_actions_l1_loss.item(),
                }
            )
        
    # Compute metrics for continuous action representations (L1 regression)
    else:
        # Get last layer hidden states
        multi_layer_hidden_states = []
        
        for item in output.hidden_states[0:]:  # 24 layers, 25 hidden_states, item = [bs,512+16+64,dim]
            # last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
            # Get hidden states for text portion of prompt+response (after the vision patches)
            # text_hidden_states = item[:, num_patches:-1]
            # Get hidden states for action portion of response
            batch_size = batch["input_ids"].shape[0]

            # actions_hidden_states = text_hidden_states[current_action_mask | next_actions_mask].reshape(batch_size, 1,NUM_TOKENS, -1).to(torch.bfloat16)  # [16,1,64,896]
            # task_latten_states = item[:, :num_patches].reshape(batch_size, 1, num_patches , -1)  # [16,1,512,896]
            
            # 不依赖label，直接取出act_states和task_states，item=[bs,len,dim], len=512+input_ids.shape[1]+64
            # Keep the original token order: visual tokens, language tokens, then action tokens.
            actions_hidden_states = item[:, -NUM_TOKENS:, :].reshape(batch_size, 1, NUM_TOKENS, -1)  # [16,1,64,d]
            task_latten_states = item[:, :num_patches].reshape(batch_size, 1, num_patches, -1)  # [16,1,512,dim]

            all_hidden_states = torch.cat((task_latten_states, actions_hidden_states),2)  # [16,1,576,dim], 576=512+64
            multi_layer_hidden_states.append(all_hidden_states)
        multi_layer_hidden_states = torch.cat(multi_layer_hidden_states, dim = 1)  # [16,25,576,dim], 576=512+64

        predicted_actions = action_head.module.predict_action(
            multi_layer_hidden_states,
            proprio=batch["proprio"] if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
            phase=cfg.phase,
            )

        loss = torch.nn.L1Loss()(predicted_actions, ground_truth_actions)

        metrics.update(
            {
                "loss_value": loss.item(),  # Detached value for logging
            }
        )

        # Get detailed L1 losses for logging
        should_log_l1_loss = use_l1_regression
        if should_log_l1_loss:
            ground_truth_curr_action = ground_truth_actions[:, 0]
            predicted_curr_action = predicted_actions[:, 0]
            ground_truth_next_actions = ground_truth_actions[:, 1:]
            predicted_next_actions = predicted_actions[:, 1:]
            curr_action_l1_loss = torch.nn.L1Loss()(ground_truth_curr_action, predicted_curr_action)
            next_actions_l1_loss = torch.nn.L1Loss()(ground_truth_next_actions, predicted_next_actions)
            if compute_diffusion_l1:
                print('curr: ',curr_action_l1_loss.item())
                # print('next: ',next_actions_l1_loss.item())

            metrics.update(
                {
                    "curr_action_l1_loss": curr_action_l1_loss.item(),
                    "next_actions_l1_loss": next_actions_l1_loss.item(),
                }
            )

    # Return both the loss tensor (with gradients) and the metrics dictionary (with detached values)
    return loss, metrics



def compute_smoothened_metrics(metrics_deques) -> dict:
    """
    Compute smoothened metrics from recent deques.

    Args:
        metrics_deques (dict): Dictionary of deques containing recent metrics.

    Returns:
        dict: Dictionary of smoothened metrics.
    """
    smoothened_metrics = {}
    for name, deque in metrics_deques.items():
        if deque and len(deque) > 0:
            smoothened_metrics[name] = sum(deque) / len(deque)
    return smoothened_metrics



def log_metrics_to_wandb(metrics, prefix, step, wandb_entity) -> None:
    """
    Log metrics to Weights & Biases.

    Args:
        metrics (dict): Dictionary of metrics to log
        prefix (str): Prefix for metric names
        step (int): Training step
        wandb_entity (str): W&B entity instance

    Returns:
        None.
    """
    log_dict = {}
    for name, value in metrics.items():
        # Map loss_value to Loss for better readability in W&B
        if name == "loss_value":
            log_dict[f"{prefix}/Loss"] = value
        # Keep other metrics as is
        else:
            log_dict[f"{prefix}/{name.replace('_', ' ').title()}"] = value
    wandb_entity.log(log_dict, step=step)



def save_training_checkpoint(
    cfg,
    run_dir,
    log_step,
    vla,
    processor,
    proprio_projector,
    prompt_projector,
    noisy_action_projector,
    action_head,
    train_dataset,
    distributed_state,
    new_state_dict,
) -> None:
    """
    Save all training checkpoints including model components, LoRA adapter, and dataset statistics.

    Args:
        cfg (FinetuneConfig): Training configuration.
        run_dir (Path): Experiment run directory path.
        log_step (int): Current logging step.
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        processor (PrismaticProcessor): OpenVLA inputs processor.
        proprio_projector (nn.Module): Proprioceptive state projector module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        action_head (nn.Module): Action head module.
        train_dataset (RLDSDataset): Training dataset.
        distributed_state (PartialState): Distributed training state.

    Returns:
        None.
    """
    # Determine checkpoint paths and naming
    if cfg.save_latest_checkpoint_only:
        checkpoint_dir = run_dir
        checkpoint_name_suffix = "latest_checkpoint.pt"
    else:
        checkpoint_dir = Path(str(run_dir) + f"--{log_step}_chkpt")
        checkpoint_name_suffix = f"{log_step}_checkpoint.pt"

    adapter_dir = checkpoint_dir / "lora_adapter"

    # Create directories and save dataset statistics (main process only)
    if distributed_state.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(adapter_dir, exist_ok=True)
        save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
        print(f"Saving Model Checkpoint for Step {log_step}")

    # Wait for directories to be created
    dist.barrier()

    # Save model components (main process only)
    if distributed_state.is_main_process:
        # Save processor and LoRA adapter
        processor.save_pretrained(checkpoint_dir)

        if cfg.use_fz:
            torch.save(vla.module.state_dict(), os.path.join(checkpoint_dir, "full_vla_model.bin"))
        else:
            vla.module.save_pretrained(adapter_dir)

        # Save other components
        if cfg.use_proprio and proprio_projector is not None:  # True
            torch.save(proprio_projector.state_dict(), checkpoint_dir / f"proprio_projector--{checkpoint_name_suffix}")

        if prompt_projector is not None:
            torch.save(prompt_projector.state_dict(), checkpoint_dir / f"prompt_projector--{checkpoint_name_suffix}")

        if cfg.use_diffusion and noisy_action_projector is not None:  # False
            torch.save(
                noisy_action_projector.state_dict(), checkpoint_dir / f"noisy_action_projector--{checkpoint_name_suffix}"
            )

        if cfg.use_l1_regression and action_head is not None:  # True
            torch.save(action_head.state_dict(), checkpoint_dir / f"action_head--{checkpoint_name_suffix}")

        if cfg.use_film:  # False
            # To be safe, just save the entire vision backbone (not just FiLM components)
            torch.save(
                vla.module.vision_backbone.state_dict(), checkpoint_dir / f"vision_backbone--{checkpoint_name_suffix}"
            )

    # Wait for model components to be saved
    dist.barrier()

    # Merge LoRA weights into base model and save resulting model checkpoint
    # Note: Can be very slow on some devices; if so, we recommend merging offline
    """
    对于lora模型，有两种保存-加载方法
    方法一（参考nova）
        直接保存被训练的模型（它是一个LoraPeftModel），直接用torch.save(model.transformer.state_dict(), main_model_path)
        加载的时候，先通过get_peft_model创建一个完全相同的模型（也是LoraPeftModel），再transformer.load_state_dict(state_dict)
    方法二（参考vla-adapter）
        保存的时候先保存lora，和方法一一样，直接用torch.save(model.transformer.state_dict(), main_model_path)
        之后再加载一个无lora的base_model
        然后合并base_model和已保存的lora，PeftModel.from_pretrained(base_vla, adapter_dir); merged_vla.merge_and_unload()
        再保存这个base_model，torch.save
        加载的时候，可以直接用base_model的from_pretrained方法来加载

    方法一对任何模型都有效，方法二最好是配合transformers模型使用，当然对于非transformers模型也是可以使用的
    """
    if cfg.use_lora and cfg.merge_lora_during_training:
        if cfg.use_minivlm:
            if distributed_state.is_main_process:
                torch.save(vla.module.state_dict(), os.path.join(checkpoint_dir, "full_vla_model.bin"))
            dist.barrier()
            return 0

        base_vla = AutoModelForVision2Seq.from_pretrained(
            cfg.config_file_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False, trust_remote_code=False
        )
        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
        merged_vla = merged_vla.merge_and_unload()

        if distributed_state.is_main_process:
            merged_vla.save_pretrained(checkpoint_dir)
            print(f"Saved merged model for Step {log_step} at: {checkpoint_dir}")

        dist.barrier()



def run_validation(
    vla,
    action_head,
    noisy_action_projector,
    proprio_projector,
    prompt_projector,
    val_dataloader,
    action_tokenizer,
    device_id,
    cfg,
    num_patches,
    log_step,
    distributed_state,
    val_time_limit,
) -> None:
    """
    Compute validation set metrics for logging.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        val_dataloader (DataLoader): Validation data loader.
        action_tokenizer (ActionTokenizer): Action tokenizer.
        device_id (str): Device ID.
        cfg (FinetuneConfig): Training configuration.
        num_patches (int): Number of vision patches.
        log_step (int): Current logging step.
        distributed_state (PartialState): Distributed training state.
        val_time_limit (int): Time limit for computing validation metrics.

    Returns:
        None.
    """
    val_start_time = time.time()
    vla.eval()
    val_batches_count = 0

    # List to store validation metrics
    all_val_metrics = []

    with torch.no_grad():
        for batch in val_dataloader:
            # Always compute L1 loss for validation, even for diffusion
            _, metrics = run_forward_pass(
                vla=vla,
                action_head=action_head,
                proprio_projector=proprio_projector,
                prompt_projector=prompt_projector,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                use_l1_regression=cfg.use_l1_regression,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=num_patches,
                compute_diffusion_l1=True,
                use_pro_version=cfg.use_pro_version
            )

            # Add the loss value to the metrics
            metrics["loss"] = metrics["loss_value"]
            all_val_metrics.append(metrics)
            val_batches_count += 1

            # Cut testing on validation set short if it exceeds time limit
            if time.time() - val_start_time > val_time_limit:
                break

    # Compute average validation metrics
    avg_val_metrics = {}
    for metric_name in all_val_metrics[0].keys():
        values = [metrics[metric_name] for metrics in all_val_metrics if metric_name in metrics]
        if values:
            avg_val_metrics[metric_name] = sum(values) / len(values)

    # Add batch count to metrics
    avg_val_metrics["val_batches_count"] = val_batches_count

    # Log validation metrics to W&B
    if distributed_state.is_main_process:
        log_metrics_to_wandb(avg_val_metrics, "VLA Val", log_step, wandb)


def load_vggt_to_vla(vggt_model, vla_model):
    """
    这个函数把vggt的24个block的qkvo线性层加载进入vla的24个layer
    另外，可以选择把24个MLP也加载
    其他参数保留来自qwen的参数
    """
    source_state_dict = vggt_model.state_dict()
    target_state_dict = vla_model.state_dict()
    new_params = {}

    for i in range(24):
        # --- 1. 处理 QKV Attention ---
        # Vision QKV weight: (3072, 1024) -> [Q(1024), K(1024), V(1024)]
        qkv_w = source_state_dict[f'aggregator.global_blocks.{i}.attn.qkv.weight']
        qkv_b = source_state_dict[f'aggregator.global_blocks.{i}.attn.qkv.bias']
        v_q_w, v_k_w, v_v_w = torch.split(qkv_w, 1024, dim=0)
        v_q_b, v_k_b, v_v_b = torch.split(qkv_b, 1024, dim=0)

        # Q: (896, 896) <- (1024, 1024)
        new_params[f'language_model.model.layers.{i}.self_attn.q_proj.weight'] = v_q_w[:896, :896]
        new_params[f'language_model.model.layers.{i}.self_attn.q_proj.bias'] = v_q_b[:896]

        # K: (128, 896) <- (1024, 1024) 截取前128个head维度
        new_params[f'language_model.model.layers.{i}.self_attn.k_proj.weight'] = v_k_w[:128, :896]
        new_params[f'language_model.model.layers.{i}.self_attn.k_proj.bias'] = v_k_b[:128]

        # V: (128, 896) <- (1024, 1024)
        new_params[f'language_model.model.layers.{i}.self_attn.v_proj.weight'] = v_v_w[:128, :896]
        new_params[f'language_model.model.layers.{i}.self_attn.v_proj.bias'] = v_v_b[:128]

        # O: (896, 896) <- (1024, 1024)
        new_params[f'language_model.model.layers.{i}.self_attn.o_proj.weight'] = source_state_dict[f'aggregator.global_blocks.{i}.attn.proj.weight'][:896, :896]

        # # --- 2. 处理 MLP (SwiGLU) ---
        # # Vision fc1 (4096, 1024) -> Lang gate/up (4864, 896)
        # v_fc1_w = source_state_dict[f'aggregator.global_blocks.{i}.mlp.fc1.weight']
        
        # for proj_name in ['gate_proj', 'up_proj']:
        #     target_key = f'language_model.model.layers.{i}.mlp.{proj_name}.weight'
        #     target_w = target_state_dict[target_key].clone()
        #     # 填充前(4096, 896)的部分
        #     target_w[:4096, :896] = v_fc1_w[:, :896]
        #     new_params[target_key] = target_w

        # # Vision fc2 (1024, 4096) -> Lang down (896, 4864)
        # v_fc2_w = source_state_dict[f'aggregator.global_blocks.{i}.mlp.fc2.weight']
        # target_down_key = f'language_model.model.layers.{i}.mlp.down_proj.weight'
        # target_down_w = target_state_dict[target_down_key].clone()
        # # 填充前(896, 4096)的部分
        # target_down_w[:896, :4096] = v_fc2_w[:896, :]
        # new_params[target_down_key] = target_down_w

    # 检查state_dict的结果
    for param_name in new_params.keys():
        if param_name not in vla_model.state_dict().keys():
            raise Exception(f"{param_name} not found in vla_model")

    # 3. 载入模型
    info = vla_model.load_state_dict(new_params, strict=False)
    print(f"############## 成功加载vggt的权重，共{len(new_params)} #################")
    return vla_model
def load_prompt_texts(task_suite: str, prompt_cache_source_path: Optional[Path] = None) -> List[str]:
    if prompt_cache_source_path is not None:
        with open(prompt_cache_source_path, "r", encoding="utf-8") as f:
            prompt_texts = json.load(f)
        if not isinstance(prompt_texts, list) or not all(isinstance(prompt, str) for prompt in prompt_texts):
            raise ValueError("prompt_cache_source_path must contain a JSON array of strings")
    else:
        suite_name = {
            "10": "libero_10",
            "goal": "libero_goal",
            "object": "libero_object",
            "spatial": "libero_spatial",
        }[task_suite]
        from libero.libero import benchmark

        benchmark_suite = benchmark.get_benchmark_dict()[suite_name]()
        prompt_texts = [benchmark_suite.get_task(task_id).language for task_id in range(benchmark_suite.n_tasks)]

    normalized_prompts = sorted({str(prompt).strip().lower() for prompt in prompt_texts if str(prompt).strip()})
    if not normalized_prompts:
        raise ValueError("No non-empty prompt text was found for the prompt cache")
    return normalized_prompts


def precompute_prompt_cache(
    all_task_prompt,
    tokenizer,
    prompt_encoder,
    cache_path,
    model_max_length,
    force_rebuild=False,
):
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_rebuild:
        return torch.load(cache_path, map_location="cpu")

    if prompt_encoder is None:
        raise ValueError("prompt_encoder is required to build the prompt cache")
    if tokenizer.pad_token_id is None:
        raise ValueError("The VLA tokenizer must define pad_token_id to build the prompt cache")

    tokenized_prompts = [
        torch.tensor(
            tokenizer(str(prompt).strip().lower(), add_special_tokens=False).input_ids[:model_max_length],
            dtype=torch.long,
        )
        for prompt in all_task_prompt
    ]
    if any(tokens.numel() == 0 for tokens in tokenized_prompts):
        raise ValueError("Prompt cache contains a prompt that tokenizes to an empty sequence")

    input_ids = pad_sequence(tokenized_prompts, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    prompt_encoder_device = next(prompt_encoder.parameters()).device
    input_ids = input_ids.to(prompt_encoder_device)
    attention_mask = attention_mask.to(prompt_encoder_device)

    with torch.no_grad():
        outputs = prompt_encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = last_token_pool(outputs.last_hidden_state, attention_mask)

    prompt_cache = {}
    for batch_idx in range(input_ids.shape[0]):
        ids_tuple = get_valid_input_ids(input_ids[batch_idx], attention_mask[batch_idx])
        prompt_cache[ids_tuple] = embeddings[batch_idx].detach().cpu().to(torch.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prompt_cache, cache_path)
    return prompt_cache


def get_prompt_cache_dim(prompt_cache: dict) -> int:
    if not prompt_cache:
        raise ValueError("Prompt cache is empty")
    return next(iter(prompt_cache.values())).shape[-1]


def load_resume_training_args(resume_path: str) -> dict:
    resume_path = Path(resume_path)
    candidates = [resume_path / "training_args.json"]
    if resume_path.name.endswith("_chkpt") and "--" in resume_path.name:
        run_dir_name = resume_path.name.rsplit("--", 1)[0]
        candidates.append(resume_path.parent / run_dir_name / "training_args.json")

    for training_args_path in candidates:
        if training_args_path.is_file():
            with open(training_args_path, "r", encoding="utf-8") as f:
                return json.load(f)

    searched_paths = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find training_args.json for resume; checked: {searched_paths}")


def validate_resume_gte_config(cfg: FinetuneConfig, resume_training_args: dict) -> None:
    previous_use_gte_encoder = resume_training_args.get("use_gte_encoder", False)
    if previous_use_gte_encoder != cfg.use_gte_encoder:
        raise ValueError(
            "Resume GTE configuration mismatch for use_gte_encoder: "
            f"checkpoint={previous_use_gte_encoder!r}, current={cfg.use_gte_encoder!r}"
        )

    if not cfg.use_gte_encoder:
        return

    current_values = {
        "prompt_encoder_path": str(cfg.prompt_encoder_path),
        "prompt_cache_path": str(cfg.prompt_cache_path),
        "prompt_cache_source_path": (
            str(cfg.prompt_cache_source_path) if cfg.prompt_cache_source_path is not None else None
        ),
        "prompt_pad_length_multiplier": cfg.prompt_pad_length_multiplier,
    }
    for field_name, current_value in current_values.items():
        if field_name not in resume_training_args:
            raise ValueError(f"Resume GTE checkpoint is missing {field_name!r} in training_args.json")
        checkpoint_value = resume_training_args[field_name]
        if checkpoint_value != current_value:
            raise ValueError(
                f"Resume GTE configuration mismatch for {field_name}: "
                f"checkpoint={checkpoint_value!r}, current={current_value!r}"
            )


def validate_resume_gte_derived_config(
    resume_training_args: dict,
    prompt_pad_length: int,
    prompt_encoder_dim: int,
) -> None:
    current_values = {
        "prompt_pad_length": prompt_pad_length,
        "prompt_encoder_dim": prompt_encoder_dim,
    }
    for field_name, current_value in current_values.items():
        if field_name not in resume_training_args:
            raise ValueError(f"Resume GTE checkpoint is missing {field_name!r} in training_args.json")
        checkpoint_value = resume_training_args[field_name]
        if checkpoint_value != current_value:
            raise ValueError(
                f"Resume GTE configuration mismatch for {field_name}: "
                f"checkpoint={checkpoint_value!r}, current={current_value!r}"
            )


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    """
    Fine-tunes base VLA on demonstration dataset via LoRA.

    Allows toggling different action representations (discrete vs. continuous), different learning objectives
    (next-token prediction vs. L1 regression vs. diffusion), FiLM. Also allows for additional model inputs,
    such as additional camera images and robot proprioceptive state. Assumes parallel action generation with
    action chunking.

    Args:
        cfg (FinetuneConfig): Training configuration.

    Returns:
        None.
    """ 

    global RAW_STATE_DICT

    if cfg.model_mode not in ("libero", "liberoplus"):
        raise ValueError("model_mode must be 'libero' or 'liberoplus'")

    prompt_projector = None
    prompt_cache = None
    prompt_pad_length = None
    prompt_encoder_dim = None

    assert not (cfg.use_l1_regression and cfg.use_diffusion), (
        "Cannot do both L1 regression and diffusion. Please pick one of them!"
    )

    # Configure the selected task suite and output paths.
    task_suite = cfg.task_suite
    assert task_suite in ["10", "goal", "object", "spatial"]
    cfg.dataset_name = f"libero_{task_suite}_no_noops"
    cfg.run_root_dir = Path(f"outputs/{task_suite}")
    if cfg.prompt_cache_path == Path("data/libero_prompt_cache.pt"):
        cfg.prompt_cache_path = Path(f"data/libero_{task_suite}_prompt_cache.pt")
    resume_training_args = None
    if cfg.resume:
        resume_training_args = load_resume_training_args(cfg.resum_vla_path)
        validate_resume_gte_config(cfg, resume_training_args)
        resume_mode = resume_training_args.get(
            "model_mode", "liberoplus" if resume_training_args.get("use_gte_encoder", False) else "libero"
        )
        if cfg.model_mode != resume_mode:
            raise ValueError(f"Resume model mode mismatch: checkpoint={resume_mode}, current={cfg.model_mode}")
    # check cfg
    assert cfg.use_lora == (not cfg.use_fz)


    # Trim trailing forward slash ('/') in VLA path if it exists
    cfg.config_file_path = cfg.config_file_path.rstrip("/")
    print(f"Fine-tuning OpenVLA Model `{cfg.config_file_path}` on `{cfg.dataset_name}`")

    # Get experiment run ID
    run_id = get_run_id(cfg)

    # Create experiment run directory
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)

    # GPU setup
    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    # Initialize wandb logging
    if distributed_state.is_main_process:
        wandb.init(project=cfg.wandb_project, name=f"ft+{run_id}", mode="offline")

    # Print detected constants
    print(
        "Detected constants:\n"
        f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n"
        f"\tACTION_DIM: {ACTION_DIM}\n"
        f"\tPROPRIO_DIM: {PROPRIO_DIM}\n"
        f"\tACTION_PROPRIO_NORMALIZATION_TYPE: {ACTION_PROPRIO_NORMALIZATION_TYPE}"
    )

    # Two options:
    # (1) Base model is on Hugging Face Hub
    #   - Then download it and record the path to the download directory
    # (2) Base model is stored locally
    #   - Then register model config in HF Auto Classes
    # In both cases, we want to check whether any changes have been made to
    # the `modeling_prismatic.py` file in this codebase; if so, we will copy
    # the file to the downloaded or locally stored checkpoint directory so
    # that the user's changes to the VLA class logic go into effect

    if model_is_on_hf_hub(cfg.config_file_path):
        # Download model directly from Hugging Face Hub
        vla_download_path = snapshot_download(repo_id=cfg.config_file_path)
        # Overwrite VLA path
        cfg.config_file_path = vla_download_path
    else:
        # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


    # Update config.json and sync model files
    if distributed_state.is_main_process:
        update_auto_map(cfg.config_file_path)
        check_model_logic_mismatch(cfg.config_file_path)

    # Wait for model files to be synced
    dist.barrier()

    # Load processor and VLA
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    processor = AutoProcessor.from_pretrained(cfg.config_file_path, trust_remote_code=True)

    if cfg.use_gte_encoder:
        assert cfg.prompt_pad_length_multiplier >= 1

        prompt_texts = load_prompt_texts(task_suite, cfg.prompt_cache_source_path)
        prompt_cache_path = Path(cfg.prompt_cache_path)
        if prompt_cache_path.exists() and not cfg.force_rebuild_prompt_cache:
            prompt_cache = precompute_prompt_cache(
                prompt_texts,
                processor.tokenizer,
                prompt_encoder=None,
                cache_path=prompt_cache_path,
                model_max_length=processor.tokenizer.model_max_length,
            )
        else:
            if distributed_state.is_main_process:
                prompt_encoder = AutoModel.from_pretrained(
                    cfg.prompt_encoder_path,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                ).to(device_id)
                prompt_encoder.eval()
                for param in prompt_encoder.parameters():
                    param.requires_grad = False

                prompt_cache = precompute_prompt_cache(
                    prompt_texts,
                    processor.tokenizer,
                    prompt_encoder,
                    cache_path=prompt_cache_path,
                    model_max_length=processor.tokenizer.model_max_length,
                    force_rebuild=True,
                )
                del prompt_encoder
                torch.cuda.empty_cache()

            dist.barrier()
            if prompt_cache is None:
                prompt_cache = precompute_prompt_cache(
                    prompt_texts,
                    processor.tokenizer,
                    prompt_encoder=None,
                    cache_path=prompt_cache_path,
                    model_max_length=processor.tokenizer.model_max_length,
                )

        max_prompt_token_length = max((len(ids_tuple) for ids_tuple in prompt_cache), default=0)
        if max_prompt_token_length == 0:
            raise ValueError("Prompt cache contains no non-empty token ID keys")
        prompt_pad_length = min(
            math.ceil(max_prompt_token_length * cfg.prompt_pad_length_multiplier),
            processor.tokenizer.model_max_length,
        )
        prompt_encoder_dim = get_prompt_cache_dim(prompt_cache)

        if resume_training_args is not None:
            validate_resume_gte_derived_config(
                resume_training_args,
                prompt_pad_length,
                prompt_encoder_dim,
            )

    if cfg.use_minivlm:
        hf_token = ''
        if 'prism-qwen25-extra-dinosiglip-224px-0_5b' in cfg.vlm_path:
            
            vlm = load(cfg.vlm_path, hf_token=hf_token, load_for_training=True)
        else:
            vlm = load_vla(
                cfg.vlm_path,
                hf_token=hf_token,
                load_for_training=True,
                )
        config = AutoConfig.from_pretrained("pretrained_models/configs/config.json")
        vla = AutoModelForVision2Seq.from_config(config, torch_dtype=torch.bfloat16).to(device_id)  # Create a new model with configuration, the parameters are randomly initialized
        # for name, param in model.named_parameters():
        #     print(f"{name}: {param.shape}")
        replace_map = [
            ("vision_backbone.dino_featurizer", "vision_backbone.featurizer"),
            ("vision_backbone.siglip_featurizer", "vision_backbone.fused_featurizer"),
            ("llm_backbone.llm", "language_model"),
            ("projector.projector.0", "projector.fc1"),
            ("projector.projector.2", "projector.fc2"),
            ("projector.projector.4", "projector.fc3"),
            ("gamma", "scale_factor"),
            ]

        def rename_state_dict_keys(state_dict, replace_map):
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k
                for old, new in replace_map:
                    if old in new_k:
                        new_k = new_k.replace(old, new)
                new_state_dict[new_k] = v
            return new_state_dict
        
        old_state_dict = vlm.state_dict()
        RAW_STATE_DICT = rename_state_dict_keys(old_state_dict, replace_map)
    
        missing_keys, unexpected_keys = vla.load_state_dict(RAW_STATE_DICT, strict=False)
        del old_state_dict

        # Attach VGGT as the language-model backbone.
        vggt_backbone = VGGTWarpperBackbone(
            embed_tokens_layer=vla.get_input_embeddings(),
            num_view=2,
            n_action_token=NUM_TOKENS,
            input_dim=vla.llm_dim,
            output_dim=1024,
            device=device_id,
            dtype=torch.bfloat16,
            model_path=os.environ.get("VGGT_MODEL_PATH", "facebook/VGGT-1B"),
        ).to(device_id).to(torch.bfloat16)
        vla.language_model = vggt_backbone

        # === 添加以下代码来更新 RAW_STATE_DICT ===
        # 1. 删除原有的 language_model 相关键
        keys_to_remove = [k for k in RAW_STATE_DICT.keys() if k.startswith('language_model.')]
        for key in keys_to_remove:
            del RAW_STATE_DICT[key]
        
        # 2. 添加 vggt_backbone 的权重到language_model下面
        vggt_state_dict = vggt_backbone.state_dict()
        for key, value in vggt_state_dict.items():
            RAW_STATE_DICT[f'language_model.{key}'] = value

    else:
        RAW_STATE_DICT ={}
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.config_file_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            trust_remote_code=False,
            ).to(device_id)

    if prompt_cache is not None:
        vla.set_prompt_cache(prompt_cache)

    # Sync CLI-controlled visual perturbation settings into the model config before wrapping with DDP/PEFT.
    vla.config.model_mode = cfg.model_mode
    vla.config.use_visual_token_perturbation = cfg.use_visual_token_perturbation
    vla.config.visual_token_pert_prob = cfg.visual_token_pert_prob
    vla.config.visual_token_pert_full_shuffle_prob = cfg.visual_token_pert_full_shuffle_prob
    vla.config.visual_token_pert_num_blocks_min = cfg.visual_token_pert_num_blocks_min
    vla.config.visual_token_pert_num_blocks_max = cfg.visual_token_pert_num_blocks_max
    vla.config.visual_token_pert_block_size_min = cfg.visual_token_pert_block_size_min
    vla.config.visual_token_pert_block_size_max = cfg.visual_token_pert_block_size_max

    # Set number of images in VLA input
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    if cfg.use_lora:
        target_modules = []
        for i in range(24):
            target_modules.extend([
                f"language_model.vggt.aggregator.frame_blocks.{i}.attn.qkv",
                f"language_model.vggt.aggregator.frame_blocks.{i}.attn.proj",
                f"language_model.vggt.aggregator.frame_blocks.{i}.mlp.fc1",
                f"language_model.vggt.aggregator.frame_blocks.{i}.mlp.fc2",
                f"language_model.vggt.aggregator.global_blocks.{i}.attn.qkv",
                f"language_model.vggt.aggregator.global_blocks.{i}.attn.proj",
                f"language_model.vggt.aggregator.global_blocks.{i}.mlp.fc1",
                f"language_model.vggt.aggregator.global_blocks.{i}.mlp.fc2",
            ])
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=2 * cfg.lora_rank,
            lora_dropout=cfg.lora_dropout,
            target_modules=target_modules,
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        for name, param in vla.named_parameters():
            if "action_queries" in name:
                param.requires_grad = True
            if name.startswith("base_model.model.projector."):
                param.requires_grad = True
            if name.startswith("base_model.model.language_model.embed_tokens"):
                param.requires_grad = True
        vla.print_trainable_parameters()

    else:
        for name, param in vla.named_parameters():
            if "action_queries" in name:
                param.requires_grad = True

    # 增加resume逻辑
    if cfg.resume:
        checkpoint = torch.load(os.path.join(cfg.resum_vla_path, "full_vla_model.bin"), map_location=vla.device)
        vla.load_state_dict(checkpoint)
        print('loaded!!!!!!!!!')

    # FiLM setup
    if cfg.use_film:
        count_parameters(vla.vision_backbone, "vla.vision_backbone (original)")
        # Wrap vision backbone with FiLM wrapper
        # Important: For this, must specify `vla.model.vision_backbone` instead of just `vla.vision_backbone`, since the
        # latter would cause the new wrapped backbone to be saved as a new attribute of `vla` instead of overwriting the
        # original one (due to the LoRA wrapper)
        vla.model.vision_backbone = FiLMedPrismaticVisionBackbone(
            vision_backbone=vla.model.vision_backbone,
            llm_dim=vla.llm_dim,
        )
        count_parameters(vla.vision_backbone, "vla.vision_backbone (post-wrap)")
        if cfg.resume:
            state_dict = load_checkpoint("vision_backbone", cfg.config_file_path, cfg.resume_step)
            vla.model.vision_backbone.load_state_dict(state_dict)
        vla.model.vision_backbone = vla.model.vision_backbone.to(device_id)

    # Wrap VLA with DDP
    vla = wrap_ddp(vla, device_id, find_unused=True)
    vla_output_dim = getattr(vla.module.language_model, 'output_dim', vla.module.llm_dim)

    if cfg.use_gte_encoder:
        prompt_projector = init_module(
            PromptProjector,
            "prompt_projector",
            cfg,
            device_id,
            {"input_dim": prompt_encoder_dim, "output_dim": vla.module.llm_dim},
            to_bf16=True,
        )

    # If applicable, instantiate proprio projector
    if cfg.use_proprio:
        proprio_projector = init_module(
            ProprioProjector,
            "proprio_projector",
            cfg,
            device_id,
            {"llm_dim": vla_output_dim, "proprio_dim": PROPRIO_DIM},
            to_bf16=True,
        )

    # If applicable, instantiate continuous action head for L1 regression
    if cfg.use_l1_regression:
        action_head = init_module(
        L1RegressionActionHead,
        "action_head",
        cfg,
        device_id,
        {
            "input_dim": vla_output_dim, 
            "hidden_dim": vla_output_dim, 
            "action_dim": ACTION_DIM,
            "use_pro_version": cfg.use_pro_version,
            },
        to_bf16=True,
        )

    # Get number of vision patches
    NUM_PATCHES = vla.module.vision_backbone.get_num_patches() * vla.module.vision_backbone.get_num_images_in_input()
    # If we have proprio inputs, a single proprio embedding is appended to the end of the vision patch embeddings

    # Instantiate optimizer
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    if cfg.use_l1_regression:
        trainable_params += [param for param in action_head.parameters() if param.requires_grad]

    if cfg.use_proprio:
        trainable_params += [param for param in proprio_projector.parameters() if param.requires_grad]
    if prompt_projector is not None:
        trainable_params += [param for param in prompt_projector.parameters() if param.requires_grad]
    print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Record original learning rate
    original_lr = optimizer.param_groups[0]["lr"]

    # Create learning rate scheduler
    # 1. MultiStepLR
    scheduler = MultiStepLR(
        optimizer,
        milestones=[cfg.num_steps_before_decay],  # Number of steps after which LR will change
        gamma=0.1,  # Multiplicative factor of learning rate decay
    )
    # 2. CosineAnnealingLR
    # scheduler = CosineAnnealingLR(
    #         optimizer,
    #         T_max=cfg.num_steps_before_decay, 
    #         eta_min=0.0001,          
    #         )

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #
    # ---
    # from prismatic.vla.datasets import DummyDataset
    #
    # train_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder,
    # )
    # ---

    # We assume that the model takes as input one third-person camera image and 1 or 2 optional wrist camera image(s)
    use_wrist_image = cfg.num_images_in_input > 1

    # Create training and optional validation datasets
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=use_wrist_image,
        use_proprio=cfg.use_proprio,
        use_minivlm=cfg.use_minivlm,
        strip_prompt_whitespace=cfg.model_mode == "liberoplus",
        )
    train_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
    if cfg.use_val_set:
        val_dataset = RLDSDataset(
            cfg.data_root_dir,
            cfg.dataset_name,
            batch_transform,
            resize_resolution=tuple(vla.module.config.image_sizes),
            shuffle_buffer_size=cfg.shuffle_buffer_size // 10,
            image_aug=cfg.image_aug,
            train=False,
        )

    # [Important] Save dataset statistics so that we can unnormalize actions during inference
    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

        # also save config
        cfg_dict = dataclasses.asdict(cfg)
        cfg_dict['data_root_dir'] = str(cfg_dict['data_root_dir'])
        cfg_dict['run_root_dir'] = str(cfg_dict['run_root_dir'])
        cfg_dict['prompt_cache_path'] = str(cfg_dict['prompt_cache_path'])
        cfg_dict['prompt_cache_source_path'] = (
            str(cfg_dict['prompt_cache_source_path']) if cfg_dict['prompt_cache_source_path'] is not None else None
        )
        cfg_dict['prompt_pad_length'] = prompt_pad_length
        cfg_dict['prompt_encoder_dim'] = prompt_encoder_dim
        with open(run_dir / "training_args.json", "w") as f:
            json.dump(cfg_dict, f, indent=4)


    # Create collator and dataloader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
        fixed_input_length=prompt_pad_length,
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important: Set to 0 if using RLDS, which uses its own parallelism
    )
    print('Len of dataloader: ', len(dataloader))
    if cfg.use_val_set:
        val_batch_size = cfg.batch_size
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,  # Important: Set to 0 if using RLDS, which uses its own parallelism
        )

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_metrics = {
        "loss_value": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
    }

    # Start training
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            # Compute training metrics and loss
            compute_diffusion_l1 = (cfg.use_l1_regression and batch_idx % cfg.diffusion_sample_freq == 0) or (cfg.use_diffusion and batch_idx % cfg.diffusion_sample_freq == 0)
            loss, metrics = run_forward_pass(
                vla=vla,
                action_head=action_head,
                proprio_projector=proprio_projector if cfg.use_proprio else None,
                prompt_projector=prompt_projector,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                use_l1_regression=cfg.use_l1_regression,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=NUM_PATCHES,
                compute_diffusion_l1=compute_diffusion_l1,
                use_pro_version=cfg.use_pro_version,
                cfg=cfg,
            )

            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps

            # Backward pass
            normalized_loss.backward()

            # Store recent train metrics
            for metric_name, value in metrics.items():
                if metric_name in recent_metrics:
                    recent_metrics[metric_name].append(value)

            # Compute gradient step index
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

            # Compute smoothened train metrics
            smoothened_metrics = compute_smoothened_metrics(recent_metrics)

            # Push Metrics to W&B (every wandb_log_freq gradient steps)
            log_step = gradient_step_idx if not cfg.resume else cfg.resume_step + gradient_step_idx
            if distributed_state.is_main_process and log_step % cfg.wandb_log_freq == 0:
                log_metrics_to_wandb(smoothened_metrics, "VLA Train", log_step, wandb)

            # [If applicable] Linearly warm up learning rate from 10% to 100% of original
            if cfg.lr_warmup_steps > 0:
                lr_progress = min((gradient_step_idx + 1) / cfg.lr_warmup_steps, 1.0)  # Cap at 1.0
                current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

            if distributed_state.is_main_process and gradient_step_idx % cfg.wandb_log_freq == 0:
                # Log the learning rate
                # Make sure to do this AFTER any learning rate modifications (e.g., warmup/decay)
                wandb.log(
                    {
                        "VLA Train/Learning Rate": scheduler.get_last_lr()[0],
                    },
                    step=log_step,
                )

            # Optimizer and LR scheduler step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress.update()

            # Save model checkpoint: either keep latest checkpoint only or all checkpoints
            if gradient_step_idx > 0 and log_step % cfg.save_freq == 0:
                save_training_checkpoint(
                    cfg=cfg,
                    run_dir=run_dir,
                    log_step=log_step,
                    vla=vla,
                    processor=processor,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    prompt_projector=prompt_projector,
                    noisy_action_projector=None,
                    action_head=action_head,
                    train_dataset=train_dataset,
                    distributed_state=distributed_state,
                    new_state_dict=RAW_STATE_DICT,
                )

            # Test model on validation set
            if cfg.use_val_set and log_step > 0 and log_step % cfg.val_freq == 0:
                run_validation(
                    vla=vla,
                    action_head=action_head,
                    noisy_action_projector=None,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    prompt_projector=prompt_projector,
                    val_dataloader=val_dataloader,
                    action_tokenizer=action_tokenizer,
                    device_id=device_id,
                    cfg=cfg,
                    num_patches=NUM_PATCHES,
                    log_step=log_step,
                    distributed_state=distributed_state,
                    val_time_limit=cfg.val_time_limit,
                )
                # Set model back to training mode after validation
                vla.train()

            # Stop training when max_steps is reached
            if log_step == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    finetune()
