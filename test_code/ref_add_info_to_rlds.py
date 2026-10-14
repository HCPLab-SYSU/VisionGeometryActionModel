"""
使用 TFDS DatasetBuilder 标准方式为 RLDS 数据集添加额外信息

这是一个标准的 TFDS Builder 实现，遵循官方推荐的方式：
1. 继承 tfds.core.GeneratorBasedBuilder
2. 实现 _info(), _split_generators(), _generate_examples()
3. 使用 builder.download_and_prepare() 自动生成数据集

使用方法：
    python test_code/add_info_to_rlds_builder.py \
        --input_path data/libero/libero_spatial_no_noops/1.0.0 \
        --output_dir data/libero_extended \
        --dataset_name libero_spatial_extended
"""

import argparse
import os
from typing import Any, Callable, Dict, Iterator

import numpy as np
import tensorflow_datasets as tfds
from tqdm import tqdm

import tensorflow as tf
# # 动态导入 tensorflow (如果已安装)
# try:
#     import tensorflow as tf
# except ImportError:
#     # 如果没有安装 tensorflow，使用 numpy dtypes 作为替代
#     # TFDS 内部会处理类型转换
#     class TFPlaceholder:
#         float32 = np.float32
#         int32 = np.int32
#         uint8 = np.uint8
#         string = str
#     tf = TFPlaceholder()


def convert_to_numpy(data):
    """
    递归地将 TensorFlow tensors 转换为 numpy arrays，外层的dict结构不变
    
    Args:
        data: 可以是 tensor, dict, list 或其他类型
        
    Returns:
        转换后的数据
    """
    if hasattr(data, 'numpy'):
        return data.numpy()
    elif isinstance(data, dict):
        return {key: convert_to_numpy(value) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(convert_to_numpy(item) for item in data)
    else:
        return data


def default_compute_extra_info(step: Dict[str, Any], step_idx: int, episode_idx: int) -> Dict[str, Any]:
    """
    默认的额外信息计算函数
    
    这是一个示例函数，你可以根据需要修改它来计算任何你想要的额外信息。
    """
    # 示例 1: 添加一个简单的标量值（如 step 索引）
    extra_scalar = float(step_idx)
    
    # 示例 2: 添加一个向量（基于 action 计算）
    action = convert_to_numpy(step['action'])
    extra_vector = np.array([action.mean(), action.std(), action.max()], dtype=np.float32)
    
    # 示例 3: 添加一个字符串标注
    extra_text = f"episode_{episode_idx}_step_{step_idx}"
    
    # 示例 4: 基于 observation 计算一些统计信息
    if 'observation' in step and 'image' in step['observation']:
        image = convert_to_numpy(step['observation']['image'])
        image_mean_brightness = float(image.mean())
    else:
        image_mean_brightness = 0.0
    
    return {
        'extra_scalar': extra_scalar,
        'extra_vector': extra_vector,
        'extra_text': extra_text,
        'image_mean_brightness': image_mean_brightness,
    }


def create_extended_features(original_features: tfds.features.FeaturesDict) -> tfds.features.FeaturesDict:
    """
    基于原始 features 创建扩展的 features（添加额外字段）
    """
    # 获取原始 step 的 features
    original_step_features = original_features['steps'].feature
    
    # 创建新的 step features，包含原有字段 + 额外字段
    extended_step_dict = {}
    
    # 复制所有原有字段
    for key, feature in original_step_features.items():
        extended_step_dict[key] = feature
    
    # 添加新字段（需要与 compute_extra_info 返回的字段对应）
    extended_step_dict['extra_scalar'] = tfds.features.Scalar(
        dtype=tf.float32,
        doc='Example extra scalar information'
    )
    extended_step_dict['extra_vector'] = tfds.features.Tensor(
        shape=(3,),
        dtype=tf.float32,
        doc='Example extra vector information'
    )
    extended_step_dict['extra_text'] = tfds.features.Text(
        doc='Example extra text information'
    )
    extended_step_dict['image_mean_brightness'] = tfds.features.Scalar(
        dtype=tf.float32,
        doc='Mean brightness of the main camera image'
    )
    
    # 创建新的 features
    extended_features = tfds.features.FeaturesDict({
        'steps': tfds.features.Dataset({
            **extended_step_dict
        }),
        'episode_metadata': original_features['episode_metadata'],
    })
    
    return extended_features


class ExtendedRLDSBuilder(tfds.core.GeneratorBasedBuilder):
    """
    扩展 RLDS 数据集的标准 TFDS Builder
    
    这个 Builder 会：
    1. 读取原始 RLDS 数据集
    2. 为每个 step 添加额外信息
    3. 使用标准的 TFDS 流程生成新数据集
    """
    
    VERSION = tfds.core.Version('1.0.0')
    
    def __init__(self, input_dataset_path: str, compute_extra_info_fn: Callable = None, **kwargs):
        self.input_dataset_path = input_dataset_path
        self.compute_extra_info_fn = compute_extra_info_fn or default_compute_extra_info
        
        original_builder = tfds.builder_from_directory(input_dataset_path)
        self.original_features = original_builder.info.features
        
        super().__init__(**kwargs)
    
    def _info(self) -> tfds.core.DatasetInfo:
        # 创建扩展的 features
        extended_features = create_extended_features(self.original_features)
        
        return tfds.core.DatasetInfo(
            builder=self,
            features=extended_features,
        )
    
    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {
            'train': self._generate_examples(),
        }
    
    def _generate_examples(self) -> Iterator[tuple]:
        """
        生成数据集的每个样本（episode）
        
        这是核心方法，负责：
        1. 读取原始数据集
        2. 为每个 step 添加额外信息
        3. yield (key, episode) 元组
        """
        
        # 加载原始数据集
        original_builder = tfds.builder_from_directory(self.input_dataset_path)
        original_dataset = original_builder.as_dataset(split='train')
        
        # 遍历每个 episode
        for episode_idx, episode in enumerate(original_dataset):
            # 处理每个 episode
            extended_steps = []
            
            for step_idx, step in enumerate(episode['steps']):
                # 将 step 转换为 numpy（保证后续处理不会有 tensor）
                step_numpy = convert_to_numpy(step)
                
                # 计算额外信息
                extra_info = self.compute_extra_info_fn(step, step_idx, episode_idx)
                
                # 合并原始数据和额外信息
                extended_step = {**step_numpy, **extra_info}
                
                extended_steps.append(extended_step)
            
            # 转换 metadata
            episode_metadata = convert_to_numpy(episode['episode_metadata'])
            
            # 构造完整的 episode
            extended_episode = {
                'steps': extended_steps,
                'episode_metadata': episode_metadata,
            }
            
            # yield (key, example) - key 必须是字符串或整数
            yield episode_idx, extended_episode


def main():
    parser = argparse.ArgumentParser(
        description='使用 TFDS Builder 标准方式为 RLDS 数据集添加额外信息'
    )
    parser.add_argument(
        '--input_path',
        type=str,
        default="data/libero/libero_spatial_no_noops/1.0.0",
        help='输入数据集路径（包含 tfrecord 和 features.json 的目录）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default="data/libero_extended",
        help='输出数据集的根目录'
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        default="libero_spatial_extended",
        help='新数据集的名称'
    )
    
    args = parser.parse_args()
    
    # 创建 Builder（注意：data_dir 会自动添加数据集名称作为子目录）
    builder = ExtendedRLDSBuilder(
        input_dataset_path=args.input_path,
        compute_extra_info_fn=default_compute_extra_info,  # 可以传入自定义函数
        data_dir=os.path.join(args.output_dir, args.dataset_name),
    )
    
    # 标准的 TFDS 流程：download_and_prepare() 会自动：
    # 1. 调用 _generate_examples() 生成数据
    # 2. 写入 TFRecord 文件
    # 3. 生成 features.json
    # 4. 生成 dataset_info.json
    # 5. 生成统计信息
    print("\n开始生成数据集（这可能需要一些时间）...")
    builder.download_and_prepare()
    
    print("\n✅ 数据集生成完成！")
    print(f"数据集位置: {builder.data_dir}")


if __name__ == '__main__':
    main()
