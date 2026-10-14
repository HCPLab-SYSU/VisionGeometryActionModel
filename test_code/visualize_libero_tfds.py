"""
利用TensorFlow Datasets自动的方式，更标准地读取libero数据集，并查看其结构
"""

import tensorflow_datasets as tfds

task_name = "spatial"
data_path = f'data/libero/libero_{task_name}_no_noops/1.0.0'

# 指向包含 tfrecord 和 features.json 的文件夹
builder = tfds.builder_from_directory(data_path)
dataset = builder.as_dataset(split='train')

for episode in dataset.take(10):

    for step_count, step in enumerate(episode['steps']):
        # step 现在是一个字典，包含了 action, observation 等
        # img = step['observation']['image'].numpy()
        action = step['action'].numpy()
        
        print(f"第 {step_count} 步, Action: {action}，尺寸为 {action.shape}")
        
        # 这里可以放置你的绘图逻辑，比如只看前几帧
        if step_count > 5: 
            break