import os
import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from peft import LoraConfig, PeftModel, get_peft_model
from experiments.robot.openvla_utils import model_is_on_hf_hub, update_auto_map, check_model_logic_mismatch, _apply_film_to_vla, _load_dataset_stats
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models import load, load_vla
from prismatic.vggt_backbone.vggt_backbone_warpper import VGGTWarpperBackbone
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
NUM_TOKENS = 64


def get_model_with_vggt(cfg, training_args):
    """
    因为加载了vggt的vla不是transformers类了，不一定支持.from_pretrained方法，所以用我的方法来创建模型
    主要做的事情包括：1.根据training_args来和training相同的方法创建模型，2.加载bin权重state_dict
    If using FiLM, wrap the vision backbone to allow for infusion of language inputs

    这部分代码拷贝自experiments/robot/openvla_utils.py: def get_vla
    然后修改了其中创建vla的相关代码
    """
    print("Instantiating pretrained VLA policy...")

    if not model_is_on_hf_hub(cfg.pretrained_checkpoint):
        # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

        # Update config.json and sync model files
        update_auto_map(cfg.pretrained_checkpoint)
        check_model_logic_mismatch(cfg.pretrained_checkpoint)

    # Build the VLA in the same way as training, then attach the VGGT backbone.
    hf_token = ''
    if 'prism-qwen25-extra-dinosiglip-224px-0_5b' in training_args["vlm_path"]:
        vlm = load(training_args["vlm_path"], hf_token=hf_token, load_for_training=True)
    else:
        vlm = load_vla(
            training_args["vlm_path"],
            hf_token=hf_token,
            load_for_training=True,
            )
    config = AutoConfig.from_pretrained("pretrained_models/configs/config.json")
    config.model_mode = training_args.get(
        "model_mode", "liberoplus" if training_args.get("use_gte_encoder", False) else "libero"
    )
    if config.model_mode not in ("libero", "liberoplus"):
        raise ValueError(f"Unknown checkpoint model mode: {config.model_mode}")
    vla = AutoModelForVision2Seq.from_config(config, torch_dtype=torch.bfloat16).to(DEVICE)
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
        device=DEVICE,
        dtype=torch.bfloat16,
        model_path=os.environ.get("VGGT_MODEL_PATH", "facebook/VGGT-1B"),
    ).to(DEVICE).to(torch.bfloat16)
    vla.language_model = vggt_backbone

    # lora part
    if training_args["use_lora"]:
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
            r=training_args["lora_rank"],
            lora_alpha=2 * training_args["lora_rank"],
            lora_dropout=training_args["lora_dropout"],
            target_modules=target_modules,
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
    # 加载ckpt
    checkpoint = torch.load(os.path.join(cfg.pretrained_checkpoint, "full_vla_model.bin"), map_location=DEVICE)
    missing, unexpected = vla.load_state_dict(checkpoint, strict=False)
    if missing:
        print(f"缺失的键: {missing[:5]}...")
    else:
        print(":::::无任何缺失的键:::::") 
    if unexpected:
        print(f"意外的键: {unexpected[:5]}...")
    else:
        print(":::::无任何意外的键:::::")
    # If using FiLM, wrap the vision backbone to allow for infusion of language inputs
    if cfg.use_film:
        vla = _apply_film_to_vla(vla, cfg)

    # Set number of images in model input
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    vla.eval()

    # Move model to device if not using quantization
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(DEVICE)

    # Load dataset stats for action normalization
    # Training receives normalized data directly; evaluation restores the saved normalization statistics.
    _load_dataset_stats(vla, cfg.pretrained_checkpoint)

    return vla
