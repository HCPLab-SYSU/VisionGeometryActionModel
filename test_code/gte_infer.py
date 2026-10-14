"""Small checks for the GTE encoder and VLA tokenizer."""
import os
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel, AutoProcessor


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """根据mask找出最后一个非padding token，作为cls token，输出为最终的embedding feature"""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


def collate_input_ids(
    input_ids_list: List[torch.Tensor], 
    model_max_length: int, 
    pad_token_id: int, 
    padding_side: str = "right"
):
    """
    来自PaddedCollatorForActionPrediction，仅对 input_ids 进行填充和截断，并生成 attention_mask。
    
    Args:
        input_ids_list: 一个包含 torch.Tensor 的列表。
        model_max_length: 模型的最大长度。
        pad_token_id: 用于填充的 token ID。
        padding_side: 填充方向，支持 "right" 或 "left"。
    """
    
    # 1. 处理填充 (Padding)
    if padding_side == "left":
        max_len = max(seq.size(0) for seq in input_ids_list)
        padded_ids = []
        for seq in input_ids_list:
            pad_len = max_len - seq.size(0)
            # 创建填充张量并拼接在左侧
            pad = torch.full((pad_len,), pad_token_id, dtype=seq.dtype, device=seq.device)
            padded_seq = torch.cat([pad, seq], dim=0)
            padded_ids.append(padded_seq)
        input_ids = torch.stack(padded_ids)
    else:
        # 默认使用 PyTorch 的 pad_sequence 进行右填充
        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id)

    # 2. 截断 (Truncate)
    # 如果填充后的长度超过了设定的最大长度，进行截断
    input_ids = input_ids[:, :model_max_length]

    # 3. 生成 Attention Mask
    # 不等于 pad_token_id 的地方为 True (1)，反之为 False (0)
    attention_mask = input_ids.ne(pad_token_id)

    return input_ids, attention_mask


def encoder_infer():
    input_texts = [
        "pick up the black bowl between the plate and the ramekin and place it on the plate",
        "pick up the black bowl from table center and place it on the plate",
        "pick up the black bowl next to the cookie box and place it on the plate"
    ]

    prompt_encoder_path = os.environ.get("PROMPT_ENCODER_PATH", "Alibaba-NLP/gte-Qwen2-1.5B-instruct")
    model = AutoModel.from_pretrained(prompt_encoder_path, trust_remote_code=True, torch_dtype=torch.float16).to("cuda:0")
    gte_tokenizer = AutoTokenizer.from_pretrained(prompt_encoder_path, trust_remote_code=True)
    gte_tokenizer.padding_side = 'right'
    vla_processor = AutoProcessor.from_pretrained('pretrained_models/configs', trust_remote_code=True)
    vla_tokenizer = vla_processor.tokenizer

    max_length = 1024

    # 第一种tokenizer
    inputs_gte = gte_tokenizer(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt', add_special_tokens=False)
    outputs_gte = model(
        input_ids=inputs_gte.input_ids.to(device="cuda:0"),
        attention_mask=inputs_gte.attention_mask.to(device="cuda:0")
        )
    embeddings_gte = last_token_pool(outputs_gte.last_hidden_state, inputs_gte['attention_mask'])  # [3,dim]

    # 第二种tokenizer
    inputs_vla = vla_tokenizer(input_texts, add_special_tokens=False)
    input_ids_list = [torch.tensor(in_ids) for in_ids in inputs_vla.input_ids]
    vla_input_ids, vla_attention_mask = collate_input_ids(
        input_ids_list=input_ids_list,
        model_max_length=vla_processor.tokenizer.model_max_length,
        pad_token_id=vla_processor.tokenizer.pad_token_id,
        padding_side="right",
        )
    outputs_vla = model(
        input_ids=vla_input_ids.to(device="cuda:0"),
        attention_mask=vla_attention_mask.to(device="cuda:0")
        )
    embeddings_vla = last_token_pool(outputs_vla.last_hidden_state, vla_attention_mask)  # [3,dim]

    # 验证两者结果相同
    flag1 = (vla_input_ids == inputs_gte.input_ids).all().item()
    flag2 = (vla_attention_mask == inputs_gte.attention_mask).all().item()
    flag3 = (outputs_gte.last_hidden_state == outputs_vla.last_hidden_state).all().item()
    flag4 = (embeddings_gte == embeddings_vla).all().item()

    # 结果很好，都是True，说明两种tokenizer和attn_mask是完全相同的

    a = 1



def main_infer():
    # Each query must come with a one-sentence instruction that describes the task
    task = 'Given a web search query, retrieve relevant passages that answer the query'
    queries = [
        get_detailed_instruct(task, 'how much protein should a female eat'),
        get_detailed_instruct(task, 'summit define')
    ]
    # No need to add instruction for retrieval documents
    documents = [
        "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
        "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments."
    ]
    input_texts = queries + documents  # list[4]

    prompt_encoder_path = os.environ.get("PROMPT_ENCODER_PATH", "Alibaba-NLP/gte-Qwen2-1.5B-instruct")
    tokenizer = AutoTokenizer.from_pretrained(prompt_encoder_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(prompt_encoder_path, trust_remote_code=True)

    max_length = 8192

    # Tokenize the input texts
    batch_dict = tokenizer(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**batch_dict)
    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])  # [4,dim]

    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)  # [4,dim]
    scores = (embeddings[:2] @ embeddings[2:].T) * 100
    print(scores.tolist())


if __name__ == "__main__":
    encoder_infer()
