"""Wrap the official VGGT aggregator with the VLA language-model interface."""
import os
import sys

import torch
from torch import nn
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

from transformers.utils import ModelOutput

# VGGT can be installed with `pip install -e`, or loaded from a source checkout.
vggt_root = os.environ.get("VGGT_ROOT")
if vggt_root and vggt_root not in sys.path:
    sys.path.insert(0, vggt_root)
from vggt.models.vggt import VGGT


@dataclass
class VGGTWrapperOutput(ModelOutput):
    """
    Copied from CausalLMOutputWithPast
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class VGGTWarpperBackbone(nn.Module):
    """
    用于代替vla的language_transformer(Qwen2ForCausalLM)的24个layer
    并完成多层hidden_state的输出

    具体来说
    用vggt/models/aggregator.py的patch_tokens = self.patch_embed(images)之后的代码
    代替transformers/models/qwen2/modeling_qwen2.py的inputs_embeds = self.embed_tokens(input_ids)之后的代码
    """
    def __init__(
        self,
            embed_tokens_layer,
            num_view=2,
            n_action_token=64,
            input_dim=896,
            output_dim=1024,
            device="cuda",
            dtype=torch.float16,
            model_path = os.environ.get("VGGT_MODEL_PATH", "facebook/VGGT-1B")
        ):
        super().__init__()
        self.embed_tokens = embed_tokens_layer
        self.vggt = VGGT.from_pretrained(model_path).to(device).to(dtype)
        self.device = device
        self.dtype = dtype

        self.num_view = num_view
        self.image_w = 224
        self.image_h = 224
        self.vision_encoder_patch_size = 14  # dino和siglip的patch_size都是14
        assert self.vision_encoder_patch_size == self.vggt.aggregator.patch_size
        self.n_action_token = n_action_token
        self.vggt_hidden_dim = 1024
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.input_projector = nn.Linear(self.input_dim, self.vggt_hidden_dim)
        if self.output_dim == 2 * self.vggt_hidden_dim:
            self.output_projector = None
        else:
            # VGGT concatenates local and global features along the hidden dimension.
            self.output_projector = nn.Linear(self.vggt_hidden_dim * 2, self.output_dim)

        # Kept for compatibility with the Hugging Face language-model interface.
        self.lm_head = nn.Linear(10, 20, bias=False)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,  # [4,592], 592 = 512+16+64，其中512是两个256的图片拼接，顺序就是512+16+64
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,  # [4,592,896], 592 = 512+16+64，dino和siglip在896的维度拼接，16是text长度不固定
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,  # True
        return_dict: Optional[bool] = None,  # True
    ) -> Union[Tuple, VGGTWrapperOutput]:
        
        # output_attentions = output_attentions if output_attentions is not None else False
        # output_hidden_states = output_hidden_states if output_hidden_states is not None else True
        # return_dict = return_dict if return_dict is not None else True

        """
        token的构造方法
        VGGT operates on frames, so language and action tokens are copied for each view.
        Their positions are appended to the image 2-D positions as a 1-D suffix.
        """

        # 先经过一个投影层，把dim对齐
        inputs_embeds = self.input_projector(inputs_embeds)  # [bs,len,896] -- [bs,len,1024]

        B = inputs_embeds.shape[0]  # batch size
        S = self.num_view  # num image, 2
        H = self.image_h
        W = self.image_w
        P = inputs_embeds.shape[1]
        C = inputs_embeds.shape[2]
        n_image_token = 256  # 512是两张图组成的，前256个是fix-camera，后256个是wrist-camera
        n_action_token = self.n_action_token
        n_language_token = P - n_action_token - n_image_token * S

        # 构造token #
        image_embeds = inputs_embeds[:, :S*n_image_token, :].reshape(B, S, n_image_token, -1)  # [bs,2,256,dim]，把512拆分为2个256
        lang_act_embeds = inputs_embeds[:, -(n_language_token+n_action_token):, :]  # [bs,16+64,dim]
        lang_act_embeds = lang_act_embeds.unsqueeze(1).expand(-1, S, -1, -1)  # [bs,2,16+64,dim]
        image_embeds = image_embeds.reshape(B*S, n_image_token, -1)  # [b*s,256,dim]
        lang_act_embeds = lang_act_embeds.reshape(B*S, n_language_token+n_action_token, -1)  # [b*s,80,dim]
        # Put language/action tokens first so their positions can be assigned as a 1-D suffix.
        tokens = torch.cat([lang_act_embeds, image_embeds], dim=1)  # [b*s,16+64+256,dim]

        # 制作frame的pos
        pos = None
        assert self.vggt.aggregator.rope is not None
        pos = self.vggt.aggregator.position_getter(
            B * S,
            H // self.vision_encoder_patch_size,
            W // self.vision_encoder_patch_size,
            device=self.device
            )  # [b*s,256,2]
        
        # 添加language和action_query的pos
        max_pos = pos.max().item()  # 获取最大位置值

        # 为语言 tokens 和可学习 tokens 分配递增位置（作为 1D 序列）
        language_pos = torch.arange(
            max_pos + 1, max_pos + 1 + n_language_token, device=self.device
            ).unsqueeze(0).expand(B*S, -1)  # [b*s,len]
        action_pos = torch.arange(
            max_pos + 1 + n_language_token, max_pos + 1 + n_language_token + n_action_token, device=self.device
            ).unsqueeze(0).expand(B*S, -1)  # [b*s,len]

        # 转换为 2D 位置格式 (y=0, x=pos_value)
        pos_language = torch.stack([torch.zeros_like(language_pos), language_pos], dim=-1).to(pos.dtype)  # [b*s,len,2]
        pos_action = torch.stack([torch.zeros_like(action_pos), action_pos], dim=-1).to(pos.dtype)  # [b*s,len,2]

        # 拼接所有位置，顺序是：lang+action+image
        pos = torch.cat([pos_language, pos_action, pos], dim=1)  # [bs*s,16+64+256,2]  顺序就是16+64+256
        assert pos.shape[0:2] == tokens.shape[0:2]  # 都是[bs*s,16+64+256,dim=2/896]
        # 成功添加language和action_query的pos

        # 重新计算一下P，因为加入了语言token和特殊token之后tokens的长度有变化
        _, P, C = tokens.shape  # [2,16+64+256,1024]

        # 开始交替frame-attn和global-attn的主要过程
        frame_idx = 0
        global_idx = 0
        hidden_state_list = []

        # frame_intermediates就是每层中间结果
        for _ in range(self.vggt.aggregator.aa_block_num):  # 24
            for attn_type in self.vggt.aggregator.aa_order:  # ["frame", "global"]
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self.vggt.aggregator._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self.vggt.aggregator._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):  # 1
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)  # [B,S,80+256,dim*2]
                hidden_state_list.append(concat_inter)

        # 组装output
        for i in range(len(hidden_state_list)):  # 24
            # 重新组织token序列，把2个256的图片合并为512，并把之前expand的lang-act-token还原（丢掉wrist-camera对应的）
            image_tokens_to_cat = [hidden_state_list[i][:, v_view, -n_image_token:, :] for v_view in range(self.num_view)]
            image_output_token = torch.cat(image_tokens_to_cat, dim=1)  # [B,256+256,dim*2]，dim*2的意思是，vggt会把local和global的特征在dim维度拼接
            lang_act_output_token = hidden_state_list[i][:, 0, :n_language_token+n_action_token, :]  # [B,80,dim*2]
            # Restore the VLA order: image tokens followed by language/action tokens.
            hidden_state_list[i] = torch.cat([image_output_token, lang_act_output_token], dim=1)  # [B,512+80,dim*2]
        
        # Repeat the first state so the wrapper exposes the expected 25 hidden-state entries.
        hidden_state_list = hidden_state_list[0:1] + hidden_state_list

        # 最后经过投影层，把dim改变
        if self.output_projector is None:
            pass
        else:
            for i in range(len(hidden_state_list)):
                hidden_state_list[i] = self.output_projector(hidden_state_list[i])

        return VGGTWrapperOutput(
            loss=None,
            logits=None,
            past_key_values=None,
            hidden_states=hidden_state_list,  # tuple[25]=[4,512+80,dim]
            attentions=None,
        )


class VGGTWarpperBackboneWithVision(nn.Module):
    """
    用于代替vla的vision_tower + language_transformer
    并完成多层hidden_state的输出
    """
    def __init__(self, device, dtype, model_path=None):
        model_path = model_path or os.environ.get("VGGT_MODEL_PATH", "facebook/VGGT-1B")
        self.vggt = VGGT.from_pretrained(model_path).to(device).to(dtype)
        self.device = device
        self.dtype = dtype
    
    def forward(self):
        pass
