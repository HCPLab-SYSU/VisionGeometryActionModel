"""
来自ai的代码，在libero环境中获取深度图
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

libero_root = os.environ.get("LIBERO_ROOT")
if libero_root and libero_root not in sys.path:
    sys.path.insert(0, libero_root)
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv


def visualize_depth(depth_map, cmap_name="viridis"):
    """
    将输入为 [H, W, 1] 或 [H, W] 的 0-1 深度图转换为 [H, W, 3] 的 RGB 图像。
    
    Args:
        depth_map (np.ndarray): 输入深度图，形状为 [256, 256, 1] 或 [256, 256]，值范围 0-1。
        cmap_name (str): Matplotlib 的颜色映射名称。
        
    Returns:
        np.ndarray: 形状为 [256, 256, 3] 的 uint8 矩阵。
    """
    if depth_map.ndim == 3:
        depth_map = np.squeeze(depth_map, axis=-1)
    
    cmap = plt.get_cmap(cmap_name)
    
    rgba_image = cmap(depth_map)
    
    rgb_image = (rgba_image[:, :, :3] * 255).astype(np.uint8)
    
    return rgb_image

def main():
    # 1. 初始化 Benchmark 和获取任务
    # 例如加载 LIBERO_100 任务集
    benchmark_dict = benchmark.get_benchmark_dict()
    suite = benchmark_dict["libero_spatial"]() 
    
    # 获取特定任务 (以第 0 个任务为例)
    task_id = 0
    task = suite.get_task(task_id)
    
    # 2. 构造 BDDL 文件的完整路径
    # 这是 LIBERO 环境加载的关键，需要拼接路径
    if not libero_root:
        raise RuntimeError("Set LIBERO_ROOT before running this helper script")
    task_bddl_file = os.path.join(
        libero_root,
        "libero/libero/bddl_files/libero_spatial/"
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl",
    )
    print(f"正在加载任务: {task.name}")

    # # 3. 准备环境参数 (env_args)
    # # 核心点：在这里设置 camera_depths=True
    # env_args = {
    #     "bddl_file_name": task_bddl_file,
    #     "camera_names": ["agentview", "robot0_eye_in_hand"], # 指定需要获取图像的相机
    #     "camera_heights": 256,
    #     "camera_widths": 256,
    #     "camera_depths": True,          # ！！！开启深度观测的关键参数
    #     "has_renderer": False,          # 离屏模式，不需要弹出 GUI
    #     "has_offscreen_renderer": True, 
    #     "use_camera_obs": True,
    #     "controller_configs": robosuite.load_controller_config(default_controller="OSC_POSE"),
    # }

    # 4. 初始化环境
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": 256, "camera_widths": 256, "camera_depths": True}
    env = OffScreenRenderEnv(**env_args)
    
    # 5. 重置并执行一个随机 Action
    obs = env.reset()
    
    # env.action_dim 通常为 7 (对于单臂 Panda 机器人)
    action = np.random.uniform(-1, 1, 7)
    obs, reward, done, info = env.step(action)

    # 7. 提取数据并进行必要的可视化翻转
    # 注意：MuJoCo 的原始渲染结果在垂直方向上是颠倒的，通常需要 [::-1] 翻转
    
    # 视角 1: AgentView (固定视角)
    agent_rgb = obs["agentview_image"][::-1]
    agent_depth_raw = obs["agentview_depth"][::-1]
    agent_depth_viz = visualize_depth(agent_depth_raw)

    # 视角 2: Robot Wrist (手眼视角)
    wrist_rgb = obs["robot0_eye_in_hand_image"][::-1]
    wrist_depth_raw = obs["robot0_eye_in_hand_depth"][::-1]
    wrist_depth_viz = visualize_depth(wrist_depth_raw)

    a = 1

    # 保存
    agent_depth_norm = (agent_depth_raw - np.min(agent_depth_raw)) / (np.max(agent_depth_raw) - np.min(agent_depth_raw))
    agent_depth_norm_vis = visualize_depth(agent_depth_norm)
    image_rgb = Image.fromarray(agent_rgb)
    image_rgb.save("rgb.png")
    image_depth = Image.fromarray(agent_depth_norm_vis)
    image_depth.save("depth.png")


if __name__ == "__main__":
    main()
