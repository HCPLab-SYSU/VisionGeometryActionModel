"""
可视化查看libero训练集的结构

实验结论：

训练集：每个suite都有10个task，每个task有若干个demo，总共每个suite大约400个demo。相同task的不同demo之间，只有物体放置位置有细微差别（细微扰动）
测试集：每个suite也是10个task，10个与训练集完全相同的task，不存在任何语言区别。相同task的不同trial之间，只有物体放置位置有细微差别（细微扰动）

spatial
    不同task的语言描述不一样，场景中有两个碗，语言会通过空间描述要求抓取其中一个碗，如“柜子上面那个碗”

object
    不同task的语言描述不一样，场景中有多个物体，语言会通过名字要求抓取其中某个物体，如“芝士盒”

goal
    不同task的语言描述不一样，场景中有多个物体，语言会要求达到不同目标，如“放入上层柜子里”或者“放入中层柜子里”

long
    不同task是不同的long-term任务，其中存在一些与上面三个相似的“易混淆”对比，比如“把芝士盒和番茄酱放入篮子”和“把汤和芝士盒放入篮子”
"""
import os
import io
import tensorflow as tf
import numpy as np
from PIL import Image
import imageio

def load_tfrecord_to_list(data_dir, prefix, max_episodes=None):
    # 1. 使用 os 获取并排序文件
    files = sorted([
        os.path.join(data_dir, f) 
        for f in os.listdir(data_dir) 
        if f.startswith(prefix)
    ])
    
    raw_dataset = tf.data.TFRecordDataset(files)
    
    # 2. 定义解析 Schema (根据你提供的 Key)
    # 注意：对于变长数组，我们先以 tf.string 或 tf.float32 读取
    feature_description = {
        'episode_metadata/file_path': tf.io.FixedLenFeature([], tf.string),
        'steps/action': tf.io.VarLenFeature(tf.float32),
        'steps/discount': tf.io.VarLenFeature(tf.float32),
        'steps/is_first': tf.io.VarLenFeature(tf.int64),
        'steps/is_last': tf.io.VarLenFeature(tf.int64),
        'steps/is_terminal': tf.io.VarLenFeature(tf.int64),
        'steps/language_instruction': tf.io.VarLenFeature(tf.string),
        'steps/observation/image': tf.io.VarLenFeature(tf.string),
        'steps/observation/joint_state': tf.io.VarLenFeature(tf.float32),
        'steps/observation/state': tf.io.VarLenFeature(tf.float32),
        'steps/observation/wrist_image': tf.io.VarLenFeature(tf.string),
        'steps/reward': tf.io.VarLenFeature(tf.float32),
    }

    structured_data = []

    print(f"开始转换数据（目标文件数: {len(files)}）...")

    for i, raw_record in enumerate(raw_dataset):
        if max_episodes is not None and i >= max_episodes:
            break
            
        # 解析原始数据
        example = tf.io.parse_single_example(raw_record, feature_description)
        
        # 构建当前 episode 的字典
        episode_dict = {}
        
        for key, value in example.items():
            # 将 SparseTensor 转为普通的 Tensor 或 numpy
            if isinstance(value, tf.SparseTensor):
                val = value.values.numpy()
            else:
                val = value.numpy()

            # --- 根据 Key 类型进行转换 ---
            
            # 情况 A: 图像数据 -> 转为 PIL.Image 对象列表
            if 'image' in key:
                # 假设每个 step 都有图，转换成 PIL 对象列表
                images = []
                for img_bytes in val:
                    images.append(Image.open(io.BytesIO(img_bytes)))
                episode_dict[key] = images
            
            # 情况 B: 文本/指令 -> 转为 str 列表
            elif 'instruction' in key or 'file_path' in key:
                if isinstance(val, np.ndarray):
                    episode_dict[key] = [v.decode('utf-8') for v in val]
                else:
                    episode_dict[key] = val.decode('utf-8')
            
            # 情况 C: 数值型数据 -> 保持为 np.array
            else:
                episode_dict[key] = val
        
        structured_data.append(episode_dict)
        
    print(f"转换完成，共加载 {len(structured_data)} 条 episode 数据。")
    return structured_data


def save_a_trajectory_video(image_list, save_file):
    writer = imageio.get_writer(save_file, fps=10)

    for pil_img in image_list:
        if pil_img.mode == 'RGBA':
            pil_img = pil_img.convert('RGB')
        img_array = np.array(pil_img)
        writer.append_data(img_array)

    writer.close()


def save_all_video(my_dataset, task_name, num_video_per_task):
    save_dir = f"rollouts/train/{task_name}"
    os.makedirs(save_dir, exist_ok=True)

    all_task_list = []
    for demo in my_dataset:
        task_prompt = demo["steps/language_instruction"][0]
        all_task_list.append(task_prompt)
    
    all_task_list = list(set(all_task_list))
    print(all_task_list)

    for the_task_prompt in all_task_list:
        counter = 0
        for demo in my_dataset:
            if counter >= num_video_per_task:
                break
            task_prompt = demo["steps/language_instruction"][0]
            if task_prompt == the_task_prompt:
                save_a_trajectory_video(demo["steps/observation/image"], f"{save_dir}/{task_prompt}_{counter}.mp4")
                counter += 1


def main():
    task_name = "spatial"
    data_path = f'data/libero/libero_{task_name}_no_noops/1.0.0' # 请修改为你的实际路径
    prefix = 'liber'

    # 加载前 5 条数据进行测试
    my_dataset = load_tfrecord_to_list(data_path, prefix, max_episodes=2000)

    save_all_video(my_dataset, task_name=task_name, num_video_per_task=3)

    a = 1


if __name__ == "__main__":
    main()