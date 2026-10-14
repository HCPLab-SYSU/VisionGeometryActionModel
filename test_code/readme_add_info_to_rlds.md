# 使用 TFDS DatasetBuilder 标准方式为 RLDS 数据集添加额外信息

## 概述

这是一个**标准的 TFDS Builder 实现**，遵循 TensorFlow Datasets 官方推荐的方式来扩展 RLDS 数据集。

### 与之前版本的区别

| 特性 | 之前的实现 (`ref_add_info_to_rlds.py`) | 新的实现 (`add_info_to_rlds_builder.py`) |
|------|----------------------------------------|------------------------------------------|
| **实现方式** | 直接使用 `tf.io.TFRecordWriter` | 继承 `tfds.core.GeneratorBasedBuilder` |
| **标准性** | 手动处理文件写入 | 完全遵循 TFDS 官方标准 |
| **元数据生成** | 手动创建 JSON 文件 | 自动生成所有元数据 |
| **使用方式** | 脚本式执行 | 标准的 Builder API |
| **可维护性** | 中等 | 高（符合 TFDS 最佳实践） |

## 核心优势

✅ **完全标准化**: 继承 `tfds.core.GeneratorBasedBuilder`，符合 TFDS 官方规范  
✅ **自动化流程**: `download_and_prepare()` 自动处理所有细节  
✅ **灵活扩展**: 通过函数参数自定义额外信息的计算逻辑  
✅ **完整元数据**: 自动生成 features.json、dataset_info.json 和统计信息  
✅ **易于集成**: 可以像使用任何 TFDS 数据集一样使用生成的数据集  

## 使用方法

### 基本用法

```bash
python test_code/add_info_to_rlds_builder.py \
    --input_path data/libero/libero_spatial_no_noops/1.0.0 \
    --output_dir data/libero_extended \
    --dataset_name libero_spatial_extended \
    --verify
```

### 参数说明

- `--input_path`: 输入 RLDS 数据集路径（包含 tfrecord 和 features.json 的目录）
- `--output_dir`: 输出数据集的根目录
- `--dataset_name`: 新数据集的名称
- `--verify`: （可选）生成完成后自动验证数据集

## 核心实现

### 1. ExtendedRLDSBuilder 类

```python
class ExtendedRLDSBuilder(tfds.core.GeneratorBasedBuilder):
    """
    扩展 RLDS 数据集的标准 TFDS Builder
    """
    
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release with extended features.',
    }
    
    def _info(self) -> tfds.core.DatasetInfo:
        """定义数据集的元信息和 features"""
        ...
    
    def _split_generators(self, dl_manager):
        """定义数据分割（train, test 等）"""
        ...
    
    def _generate_examples(self) -> Iterator[tuple]:
        """生成数据集的每个样本（episode）"""
        ...
```

### 2. 自定义额外信息

只需修改两个地方：

#### A. 计算额外信息的函数

```python
def compute_extra_info(step, step_idx, episode_idx):
    """
    ⭐⭐⭐ 在这里自定义要添加的额外信息 ⭐⭐⭐
    """
    # 示例：添加 step 索引
    extra_scalar = float(step_idx)
    
    # 示例：基于 action 计算统计量
    action = convert_to_numpy(step['action'])
    extra_vector = np.array([action.mean(), action.std()], dtype=np.float32)
    
    # 示例：添加文本标注
    extra_text = f"episode_{episode_idx}_step_{step_idx}"
    
    return {
        'extra_scalar': extra_scalar,
        'extra_vector': extra_vector,
        'extra_text': extra_text,
    }
```

#### B. 定义字段类型

```python
def create_extended_features(original_features):
    """
    ⭐⭐⭐ 在这里定义新字段的类型 ⭐⭐⭐
    """
    extended_step_dict['extra_scalar'] = tfds.features.Scalar(
        dtype=tf.float32,
        doc='Step index as scalar'
    )
    extended_step_dict['extra_vector'] = tfds.features.Tensor(
        shape=(2,),  # 修改为你的向量维度
        dtype=tf.float32,
        doc='Action statistics'
    )
    extended_step_dict['extra_text'] = tfds.features.Text(
        doc='Custom annotation'
    )
    ...
```

## 支持的数据类型

### 标量 (Scalar)
```python
tfds.features.Scalar(dtype=tf.float32, doc='Description')
tfds.features.Scalar(dtype=tf.int32, doc='Description')
```

### 向量/矩阵 (Tensor)
```python
tfds.features.Tensor(shape=(n,), dtype=tf.float32, doc='Description')
tfds.features.Tensor(shape=(h, w), dtype=tf.float32, doc='Description')
```

### 字符串 (Text)
```python
tfds.features.Text(doc='Description')
```

### 图像 (Image)
```python
tfds.features.Image(
    shape=(height, width, channels), 
    dtype=tf.uint8,
    encoding_format='jpeg'
)
```

## 加载生成的数据集

生成完成后，有两种方式加载数据集：

### 方法 1: 从 data_dir 加载

```python
import tensorflow_datasets as tfds

builder = tfds.builder('libero_spatial_extended', 
                       data_dir='data/libero_extended')
dataset = builder.as_dataset(split='train')

# 使用数据集
for episode in dataset:
    for step in episode['steps']:
        action = step['action'].numpy()
        extra_scalar = step['extra_scalar'].numpy()
        extra_vector = step['extra_vector'].numpy()
        # ... 使用新增的字段
```

### 方法 2: 从目录加载

```python
import tensorflow_datasets as tfds

dataset_path = 'data/libero_extended/libero_spatial_extended/1.0.0'
builder = tfds.builder_from_directory(dataset_path)
dataset = builder.as_dataset(split='train')
```

## 工作流程

```
┌─────────────────────────────────────────────────────────────┐
│  1. 创建 ExtendedRLDSBuilder                                 │
│     - 指定输入数据集路径                                      │
│     - 指定输出目录和数据集名称                                │
│     - 传入自定义的 compute_extra_info_fn (可选)              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  2. 调用 builder.download_and_prepare()                      │
│     自动执行以下步骤：                                        │
│     a. 调用 _info() 获取数据集元信息                         │
│     b. 调用 _split_generators() 确定数据分割                 │
│     c. 调用 _generate_examples() 生成数据                    │
│     d. 写入 TFRecord 文件                                    │
│     e. 生成 features.json                                    │
│     f. 生成 dataset_info.json                               │
│     g. 计算数据集统计信息                                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  3. 使用 builder.as_dataset() 加载数据集                      │
│     或使用 tfds.builder_from_directory() 从目录加载           │
└─────────────────────────────────────────────────────────────┘
```

## 输出结构

生成的数据集具有以下目录结构：

```
data/libero_extended/
└── libero_spatial_extended/
    └── extended_rlds_builder/
        └── 1.0.0/
            ├── dataset_info.json          # 数据集元信息
            ├── features.json              # Feature 定义
            ├── extended_rlds_builder-train.tfrecord-*  # 数据文件
            └── dataset_statistics_*.json  # 统计信息
```

## 完整示例

```python
#!/usr/bin/env python
"""
自定义示例：为 LIBERO 数据集添加深度图信息
"""

import numpy as np
from test_code.add_info_to_rlds_builder import ExtendedRLDSBuilder, convert_to_numpy

def compute_depth_info(step, step_idx, episode_idx):
    """计算深度相关的额外信息"""
    # 假设你有一个函数可以从图像计算深度
    if 'observation' in step and 'image' in step['observation']:
        image = convert_to_numpy(step['observation']['image'])
        
        # 这里添加你的深度计算逻辑
        # depth_map = compute_depth(image)
        # 为了示例，我们只是计算图像的灰度统计
        gray = image.mean(axis=-1)  # 简化的"深度"表示
        
        return {
            'depth_mean': float(gray.mean()),
            'depth_std': float(gray.std()),
        }
    else:
        return {
            'depth_mean': 0.0,
            'depth_std': 0.0,
        }

# 创建 Builder
builder = ExtendedRLDSBuilder(
    input_dataset_path='data/libero/libero_spatial_no_noops/1.0.0',
    compute_extra_info_fn=compute_depth_info,  # 使用自定义函数
    data_dir='data/libero_with_depth',
)

# 生成数据集
builder.download_and_prepare()

# 使用数据集
dataset = builder.as_dataset(split='train')
for episode in dataset.take(1):
    for step in episode['steps'].take(5):
        print(f"Depth mean: {step['depth_mean'].numpy()}")
        print(f"Depth std: {step['depth_std'].numpy()}")
```

## 注意事项

1. **内存管理**: Builder 会逐个 episode 处理数据，避免内存溢出
2. **数据类型一致性**: 确保 `compute_extra_info()` 返回的数据类型与 `create_extended_features()` 中定义的类型一致
3. **Tensor 转换**: 所有 TensorFlow tensors 会被自动转换为 numpy arrays
4. **原始数据完整性**: 原始数据集不会被修改，新数据集保存在指定的输出路径

## 故障排除

### 问题 1: "TypeError: DatasetBuilder.__init__() got an unexpected keyword argument 'name'"

**原因**: TFDS Builder 不接受 `name` 参数  
**解决**: 数据集名称通过类名自动确定，或者在 data_dir 中指定

### 问题 2: "Error encoding: tf.Tensor"

**原因**: 未将 TensorFlow tensors 转换为 numpy arrays  
**解决**: 使用 `convert_to_numpy()` 函数

### 问题 3: "Shape mismatch"

**原因**: 返回的数据 shape 与定义的不一致  
**解决**: 检查 `create_extended_features()` 中的 shape 定义

## 参考资料

- [TensorFlow Datasets 官方文档](https://www.tensorflow.org/datasets)
- [RLDS 文档](https://github.com/google-research/rlds)
- [添加自定义数据集](https://www.tensorflow.org/datasets/add_dataset)
- [GeneratorBasedBuilder API](https://www.tensorflow.org/datasets/api_docs/python/tfds/core/GeneratorBasedBuilder)

## 版本历史

- **v1.0** (2026-03-20): 初始版本，基于 TFDS DatasetBuilder 的标准实现
