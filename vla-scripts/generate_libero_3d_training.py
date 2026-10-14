"""
用这个代码为libero生成深度图和相机内外参

现在生成深度图的代码是这样：取bning的libero_mask_depth数据集，然后根据第一个state的匹配来找到对应的depth，后期需要改成按照第一个image来匹配
"""
import os
import sys
import numpy as np
import json
import gc
# from libero.libero import benchmark
# from libero.libero.envs import OffScreenRenderEnv
# from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix

from datasets import load_dataset
from collections import defaultdict
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Any, Callable, Dict, Iterator
from PIL import Image
import time
import pickle


def get_a_camera_param(bddl_path):
    # 创建环境（只开启固定相机 agentview）
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview"],
        camera_heights=256,
        camera_widths=256,
    )

    # 固定相机 agentview
    # 内参矩阵：描述相机如何将3D点投影到256×256图像
    K = get_camera_intrinsic_matrix(env.sim, "agentview", 256, 256)
    # 内参逆矩阵：用于反投影，从2D图像坐标恢复3D方向
    K_inv = np.linalg.inv(K)
    # 相机->世界变换矩阵：将相机坐标系的点转换到世界坐标系，前3×3(R)相机坐标轴在世界坐标系中的方向，最后一列(t)相机原点的世界坐标位置
    T = get_camera_extrinsic_matrix(env.sim, "agentview")
    # 世界->相机变换矩阵：投影时使用，将世界坐标的3D点转到相机坐标系
    T_inv = np.linalg.inv(T)

    cameras_param_dict = {
        "camera_name": "agentview",
        "type": "fixed",
        "bddl_file": bddl_path,
        "K": K.tolist(),
        "K_inv": K_inv.tolist(),
        "T_cam_to_world": T.tolist(),
        "T_world_to_cam": T_inv.tolist(),
    }

    env.close()

    return cameras_param_dict



def get_camera_param():
    bddl_root = os.environ.get("LIBERO_BDDL_ROOT", "libero/libero/bddl_files")
    cameras = {}

    for task_suite_name in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:

        cameras[task_suite_name] = {}

        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[task_suite_name]()
        num_tasks = task_suite.n_tasks

        for task_idx in range(num_tasks):
            task_name = task_suite.get_task(task_idx).name

            bddl_file = os.path.join(bddl_root, task_suite_name, task_name + ".bddl")
            assert os.path.isfile(bddl_file)

            cameras_param_dict = get_a_camera_param(bddl_file)

            cameras[task_suite_name][task_name] = cameras_param_dict

    # save
    with open("libero_cameras_params.json", "w") as f:
        json.dump(cameras, f, indent=4)



def check_lenghth_in_bning_dataset():
    """
    检查这个xxx_lerobot_mask_depth数据集的metadata的lenght是不是正确的

    经过检查，全部4个suites的长度都是正确的，全部4个suites都能通过检查
    """
    mask_depth_dataset_root = os.environ.get(
        "LIBERO_MASK_DEPTH_ROOT", "data/{task_suite_name}_lerobot_mask_depth"
    )
    episode_mapping_file = os.environ.get(
        "LIBERO_MASK_DEPTH_EPISODES",
        "data/{task_suite_name}_lerobot_mask_depth/meta/episodes.jsonl",
    )

    for task_suite_name in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
        mask_depth_dataset = mask_depth_dataset_root.format(task_suite_name=task_suite_name)
        episode_mapping = load_jsonl_file(episode_mapping_file.format(task_suite_name=task_suite_name))  # list[dict]

        mapping = {}
        for item in episode_mapping:
            episode_index = item["episode_index"]
            length = item["length"]
            assert episode_index not in mapping.keys()
            mapping.update({str(episode_index): length})

        ds = load_dataset(mask_depth_dataset)["train"]

        organized = ds_to_organized(ds)

        # 节省内存
        del ds # 现在可以安全删除了
        gc.collect()

        for task_index in organized.keys():
            for episode_index in organized[task_index].keys():
                length_a = len(organized[task_index][episode_index])
                length_b = mapping[str(episode_index)]
                assert length_a == length_b

        print(f"=================== {task_suite_name} 检查通过 =================")

        del organized # 现在可以安全删除了
        gc.collect()


"""
以下是resave_libero_depthmap的代码
* 首先需要下载libero_object_lerobot_mask_depth
* 上述数据集会被读取并转换为list，同时读取libero_noop数据集，按照长度和state匹配两者
* 然后把depth_map放在libero_noop数据集，重新保存为rlds数据集
"""
class ExtraInfoContainer:
    """包装外部数据集和计算逻辑"""
    
    def __init__(self, external_dataset, episode_mapping):
        self.external_dataset = external_dataset
        self.episode_mapping = episode_mapping
    
    def __call__(self, episode, episode_i):
        # 访问外部数据集
        organized = self.external_dataset
        episode_mapping = self.episode_mapping
    
        # 找到这个episode在organized里面对应的index
        matched_task_index, matched_episode_index = match_episode(organized, episode, episode_mapping, episode_i)

        extra_info_list = []  # extra_info_list[step_idx] = {}
        for step_idx, step in enumerate(episode['steps']):  # 标准遍历方法

            # 从organized中获取信息
            the_step_idx = min(step_idx, max(organized[matched_task_index][matched_episode_index].keys()))  # 防止step_idx超出索引范围
            sample = organized[matched_task_index][matched_episode_index][the_step_idx]
            assert matched_task_index == sample["task_index"]
            assert matched_episode_index == sample["episode_index"]
            assert the_step_idx == sample["frame_index"]
            image_depth = sample["observation.images.image_depth"]  # 这里还是PIL
            wrist_depth = sample["observation.images.wrist_depth"]  # 这里还是PIL

            extra_info = {
                'image_depth': np.array(image_depth.convert("L"))[:, :, None],  # to L to np
                'wrist_depth': np.array(wrist_depth.convert("L"))[:, :, None],  # to L to np
            }
            extra_info_list.append(extra_info)
        
        return extra_info_list


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
    extended_step_dict['image_depth'] = tfds.features.Image(
        shape=(256, 256, 1),
        dtype=tf.uint8,
        encoding_format='jpeg',
        doc='Fixed camera depth observation.',
        )
    extended_step_dict['wrist_depth'] = tfds.features.Image(
        shape=(256, 256, 1),
        dtype=tf.uint8,
        encoding_format='jpeg',
        doc='Wrist camera depth observation.',
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
    
    def __init__(
            self,
            input_dataset_path: str,
            dataset_name: str,
            extending_info_fn: Callable,
            **kwargs):
        self.input_dataset_path = input_dataset_path
        self._custom_name = dataset_name
        self.extending_info_fn = extending_info_fn
        
        original_builder = tfds.builder_from_directory(input_dataset_path)
        self.original_features = original_builder.info.features
        
        super().__init__(**kwargs)

    @property
    def name(self):
        """覆盖默认的数据集名称"""
        return self._custom_name
    
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
        # original_dataset = original_builder.as_dataset(split='train[11:13]')
        
        # 遍历每个 episode
        for episode_idx, episode in enumerate(original_dataset):
            # 处理每个 episode
            extended_steps = []

            extra_info_list = self.extending_info_fn(episode, episode_idx)
            
            for step_idx, step in enumerate(episode['steps']):
                # 将 step 转换为 numpy（保证后续处理不会有 tensor）
                step_numpy = convert_to_numpy(step)
                
                # 计算额外信息
                extra_info = extra_info_list[step_idx]
                
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


def resave_libero_depthmap():
    mask_depth_dataset_root = os.environ.get(
        "LIBERO_MASK_DEPTH_ROOT", "data/{task_suite_name}_lerobot_mask_depth"
    )
    episode_mapping_file = os.environ.get(
        "LIBERO_MASK_DEPTH_EPISODES",
        "data/{task_suite_name}_lerobot_mask_depth/meta/episodes.jsonl",
    )
    save_dir = os.environ.get("LIBERO_WITH_MASK_DEPTH_ROOT", "data/libero_with_mask_depth")
    os.makedirs(save_dir, exist_ok=True)
    original_libero_data_root = os.environ.get(
        "MODIFIED_LIBERO_RLDS_ROOT", "data/modified_libero_rlds/{task_suite_name}_no_noops/1.0.0"
    )

    # 把organized结果保存为临时文件
    temp_file_dir = os.environ.get("LIBERO_TEMP_DIR", "data/temp")
    os.makedirs(temp_file_dir, exist_ok=True)
    use_temp_cache = True

    for task_suite_name in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
    # for task_suite_name in ["libero_spatial"]:
        print(f"\n\n================处理 {task_suite_name} 数据集===================\n\n")
        start_time = time.time()

        mask_depth_dataset = mask_depth_dataset_root.format(task_suite_name=task_suite_name)
        original_dataset_path = original_libero_data_root.format(task_suite_name=task_suite_name)
        episode_mapping = load_jsonl_file(episode_mapping_file.format(task_suite_name=task_suite_name))  # list[dict]

        temp_cache_file = os.path.join(temp_file_dir, f"{task_suite_name}.pkl")
        if use_temp_cache and (os.path.isfile(temp_cache_file[:-4] + "_key_0.pkl")):
            # 从临时缓存文件中直接读取
            organized = chunked_pickle_load(temp_cache_file)

        else:
            ds = load_dataset(mask_depth_dataset)["train"]
            # ds = load_dataset(mask_depth_dataset)["train"].select(range(6000))
            organized = ds_to_organized(ds)

            del ds
            gc.collect()

            if use_temp_cache:
                # 保存到临时缓存文件中
                chunked_pickle_dump(organized, temp_cache_file)

        print(f"\n\n=========={task_suite_name} 第一阶段完成，将lerobot数据整理为dict==========\n\n")

        extender = ExtraInfoContainer(
            external_dataset=organized,
            episode_mapping=episode_mapping,
            )

        builder = ExtendedRLDSBuilder(
            input_dataset_path=original_dataset_path,
            dataset_name=task_suite_name,
            extending_info_fn=extender,  # 可以传入自定义函数
            data_dir=save_dir,
        )

        builder.download_and_prepare()

        end_time = time.time()
        hours = round((end_time - start_time) / 3600, 2)

        print(f"\n\n================成功 {task_suite_name} 数据集===================")
        print(f"\n保存到{os.path.join(save_dir, task_suite_name)}")
        print(f"\n用时{hours}小时\n\n")

        del organized
        gc.collect()


def chunked_pickle_dump(large_obj, pickle_file):
    # 逐个处理顶层key，降低内存占用
    for idx, (key, value) in enumerate(large_obj.items()):
        chunk_file = pickle_file[:-4] + f"_key_{key}.pkl"
        with open(chunk_file, "wb") as f:
            pickle.dump({key: value}, f)


def chunked_pickle_load(pickle_file):
    # 逐个加载分块并合并
    large_obj = {}
    meta_file_name = os.path.basename(pickle_file)[:-4]
    file_list_all = os.listdir(os.path.dirname(pickle_file))
    file_list = [fff for fff in file_list_all if meta_file_name in fff]

    for filename in file_list:
        key = int(filename[:-4].split("_key_")[-1])

        with open(os.path.join(os.path.dirname(pickle_file), filename), "rb") as f:
            chunk_data = pickle.load(f)
            large_obj.update({key: chunk_data[key]})
    
    return large_obj


def load_jsonl_file(file_path):
    all_lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_object = json.loads(line.strip())
            all_lines.append(json_object)
    return all_lines


def ds_to_organized(ds):
    """
    把lerobot数据集整理为list，原本lerobot数据集是展平的，现在按照index重新整理

    经过检查，原本的ds中的episode_index是完全连续排列的，完全没有重复的（见episode_index.json），意思是是task2的第一个episode_index是接着task1来的
    而ds_to_organized能很好地兼容这件事，也就是说原本episode_index是多少，在新组织的dict中它的key就是多少，不是按照顺序从0开始的
    """
    organized = {}
    
    for sample in ds:
        t, e, f = sample["task_index"], sample["episode_index"], sample["frame_index"]
        
        # 逐层初始化，避免使用巨大的中间 defaultdict
        if t not in organized: organized[t] = {}
        if e not in organized[t]: organized[t][e] = {}
        
        # 直接存储，不进行多余的深拷贝，只保留这些key
        organized[t][e][f] = {
            "task_index": t,
            "episode_index": e,
            "frame_index": f,
            "observation.state": sample["observation.state"],
            "observation.images.image_depth": sample["observation.images.image_depth"],
            "observation.images.wrist_depth": sample["observation.images.wrist_depth"],
        }
    
    # 此时 ds 依然存在，但我们已经完成了引用分配
    return organized


def remove_prompt_prefix(task_prompt):
    # 移除libero_10的prompt前缀
    prefix_list = [
        "KITCHEN SCENE3 ",
        "KITCHEN SCENE4 ",
        "KITCHEN SCENE6 ",
        "KITCHEN SCENE8 ",
        "LIVING ROOM SCENE1 ",
        "LIVING ROOM SCENE2 ",
        "LIVING ROOM SCENE2 ",
        "LIVING ROOM SCENE5 ",
        "LIVING ROOM SCENE6 ",
        "STUDY SCENE1 ",
        ]
    for prefix in prefix_list:
        if prefix in task_prompt:
            task_prompt = task_prompt.split(prefix)[-1]
            break
    return task_prompt


def match_episode(organized, episode, episode_mapping, episode_i):
    """
    匹配代码的写法：先for 10个，匹配task_index，再遍历100个轨迹，找到长度相同的，生成一个1*10的矩阵，然后计算state与矩阵的均方差找到最优匹配

    organized[task_index][episode_index][step_index]
    episode: step = next(iter(episode["steps"]))  step['observation']['image']
    """

    first_step = next(iter(episode["steps"]))

    # 收集关键信息
    target_prompt = first_step["language_instruction"].numpy().decode("utf-8")
    target_len = len(episode["steps"])
    target_first_state = first_step["observation"]["state"].numpy()  # [8]

    # 找到当前的episode数据（来自noop）对应哪个task_index
    matched_task_index = None
    for task_index in organized.keys():
        # 找这个task_index对应的task_instruction
        episode_index = list(organized[task_index].keys())[0]
        task_prompt = None
        for mapping_item in episode_mapping:
            if mapping_item["episode_index"] == episode_index:
                task_prompt = mapping_item["tasks"][0]
                break
        assert task_prompt is not None, f"No matched task_prompt for episode_i = {episode_i}"

        # 判断这个是否是这个task_index
        if remove_prompt_prefix(task_prompt) == target_prompt:
            matched_task_index = task_index
            break
    
    assert matched_task_index is not None, f"No matched task_index for episode_i = {episode_i}"

    # 找到当前的episode数据（来自noop）对应哪个episode_index
    stored_first_state = []
    stored_first_state_index = []
    for episode_index in organized[matched_task_index].keys():
        episode_len = len(organized[matched_task_index][episode_index])
        if episode_len != target_len:
            continue
        stored_first_state.append(organized[matched_task_index][episode_index][0]["observation.state"])
        stored_first_state_index.append(episode_index)

    # assert len(stored_first_state) > 0, f"No matched episode_length for episode_i = {episode_i}"
    if len(stored_first_state) > 0:
        # 这是正常的
        pass
    else:
        # 结果不正常，死马当活马医，把全部50条轨迹都放进去
        print(f"No matched episode_length for episode_i = {episode_i}")
        for episode_index in organized[matched_task_index].keys():
            stored_first_state.append(organized[matched_task_index][episode_index][0]["observation.state"])
            stored_first_state_index.append(episode_index)
    
    stored_first_state = np.stack(stored_first_state, axis=0)  # [n, 8]
    mse_per_row = ((stored_first_state - target_first_state)**2).mean(axis=1)  # [n]
    min_value = np.min(mse_per_row)
    min_index = np.argmin(mse_per_row)

    print(f"最优匹配的差距为: {min_value}")

    matched_episode_index = stored_first_state_index[min_index]

    return matched_task_index, matched_episode_index



if __name__ == "__main__":
    # get_camera_param()
    # check_lenghth_in_bning_dataset()
    resave_libero_depthmap()