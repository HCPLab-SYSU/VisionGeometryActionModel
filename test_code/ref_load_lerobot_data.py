"""
加载binhng/libero_object_lerobot_mask_depth数据集的代码
"""
import torch
import numpy as np
from PIL import Image
import os
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from datasets import load_dataset
from collections import defaultdict

MASK_DEPTH_DATASET_ROOT = os.environ.get(
    "LIBERO_MASK_DEPTH_ROOT", "data/libero_object_lerobot_mask_depth"
)


def main():

    ds = load_dataset(MASK_DEPTH_DATASET_ROOT)["train"]

    # 1. 先按 task → episode 收集（用 defaultdict 避免手动判断）
    temp = defaultdict(lambda: defaultdict(dict))

    for sample in ds:
        task = sample["task_index"]
        episode = sample["episode_index"]
        frame_idx = sample["frame_index"]          # 每个 episode 内从 0 开始的步序号
        temp[task][episode][frame_idx] = sample

    # 2. 转成普通 dict，并确保每一层都按数字升序（task、episode、frame_index）
    organized = {}
    for task in sorted(temp.keys()):                     # task 0→9
        organized[task] = {}
        for episode in sorted(temp[task].keys()):        # 同一 task 下的 episode
            # frame_index 已作为 key，天然按从小到大排列（因为我们后面用 sorted）
            organized[task][episode] = {
                frame: temp[task][episode][frame]
                for frame in sorted(temp[task][episode].keys())
            }

    # ====================== 使用示例 ======================
    print(f"整理完成！结构为：organized[task][episode][frame_index] = sample")
    print(f"organized的数据类型为: {type(organized)}")
    print(f"总 task: {len(organized)}")
    print(f"organized的key为: {organized.keys()}")
    print(f"task 0 的 episode 数量: {len(organized[0])}")
    print(f"task 0 的 key: {list(organized[0].keys())[0:5]}")
    print(f"task 0 - episode 0 的步数: {len(organized[0][0])}")

    # 访问方式（超级直观）
    sample = organized[0][0][2]          # ← task=1, episode=3, 第5步
    print(f"拿到 task1-ep3-frame5 的图像类型: {type(sample['observation.images.image'])}")


def main_lerobot():
    """
    这个代码是lerobot的官方读取数据的代码（来自AI）
    不过，这个代码运行会报错：We introduced a new format since v3.0 which is not backward compatible with v2.1.

    解决方法，可以用lerobot的官方代码把v2.1的数据集转换为v3.0，大概是
    python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 --repo-id="$LIBERO_MASK_DEPTH_ROOT"
    """
    # 加载数据集（自动使用你已下载的本地缓存，或从 HF 下载）
    # repo_id 就是你在 HF 下载的那个数据集名称
    dataset = LeRobotDataset(MASK_DEPTH_DATASET_ROOT)

    # 取前 5 个数据（注意：LeRobotDataset 是按 frame 索引，不是 episode）
    # 每个 sample 就是一个 dict，里面包含所有观测、动作、深度图等（已转为 torch.Tensor）
    first_five: list[dict] = [dataset[i] for i in range(5)]

    # ====================== 验证 & 打印示例 ======================
    print(f"成功转换为 list[dict]，长度: {len(first_five)}")
    print(f"每个 dict 的键（结构化后的 LeRobot 格式）:")
    print(list(first_five[0].keys()))

    # 示例：查看第一个数据的形状（RGB、depth、mask 等）
    sample0 = first_five[0]
    print("\n第一个数据的关键字段形状示例：")
    print(f"  - observation.images.image:          {sample0['observation.images.image'].shape}")      # (3, 256, 256)
    print(f"  - observation.images.image_depth:    {sample0['observation.images.image_depth'].shape}") # (1, 256, 256) 或 (256, 256)
    print(f"  - observation.images.image_mask:     {sample0['observation.images.image_mask'].shape}")
    print(f"  - observation.state:                 {sample0['observation.state'].shape}")             # (8,)
    print(f"  - action:                            {sample0['action'].shape}")                        # (7,)
    print(f"  - episode_index:                     {sample0['episode_index']}")

    # 如果你想把 tensor 转成 numpy（方便后续处理或保存）：
    import numpy as np
    first_five_numpy = []
    for sample in first_five:
        sample_np = {
            k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
            for k, v in sample.items()
        }
        first_five_numpy.append(sample_np)

    print("\n已额外生成 numpy 版本（first_five_numpy），长度一样")



if __name__ == "__main__":
    main()
