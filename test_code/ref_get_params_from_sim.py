"""
从robosuite仿真器直接获取LIBERO相机参数
"""
import sys
sys.path.insert(0, '/home/lqc/LIBERO')

import numpy as np
import json
from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix
from libero.libero.envs import OffScreenRenderEnv


# 任务集合的 bddl 文件
bddl_files = {
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket": "/home/lqc/LIBERO/libero/libero/bddl_files/libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket.bddl",
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket": "/home/lqc/LIBERO/libero/libero/bddl_files/libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket.bddl",
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket": "/home/lqc/LIBERO/libero/libero/bddl_files/libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket.bddl",
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate": "/home/lqc/LIBERO/libero/libero/bddl_files/libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate.bddl",
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate": "/home/lqc/LIBERO/libero/libero/bddl_files/libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate.bddl",
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy": "/home/lqc/LIBERO/libero/libero/bddl_files/libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.bddl",
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it": "/home/lqc/LIBERO/libero/libero/bddl_files/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl",
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it": "/home/lqc/LIBERO/libero/libero/bddl_files/libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it.bddl",
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it": "/home/lqc/LIBERO/libero/libero/bddl_files/libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it.bddl",
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove": "/home/lqc/LIBERO/libero/libero/bddl_files/libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove.bddl",
}

cameras = {}

for name, bddl_path in bddl_files.items():
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

    cameras[name] = {
        "camera_name": "agentview",
        "type": "fixed",
        "bddl_file": bddl_path,
        "K": K.tolist(),
        "K_inv": K_inv.tolist(),
        "T_cam_to_world": T.tolist(),
        "T_world_to_cam": T_inv.tolist(),
    }

    env.close()

with open("/home/lqc/par_data/libero_10_cameras_params.json", "w") as f:
    json.dump(cameras, f, indent=2)



# 手腕相机 robot0_eye_in_hand
# 内参矩阵：描述相机如何将3D点投影到256×256图像
# K_wrist = get_camera_intrinsic_matrix(env.sim, "robot0_eye_in_hand", 256, 256)
# # 内参逆矩阵：用于反投影，从2D图像坐标恢复3D方向
# K_wrist_inv = np.linalg.inv(K_wrist)

# cameras["robot0_eye_in_hand"] = {
#     'type': 'dynamic',
#     'K': K_wrist.tolist(),
#     'K_inv': K_wrist_inv.tolist(),
# }

# ----------------------------------------------------------------------------------------------------
# 手腕相机外参每帧变化，使用时需实时获取，实际使用示例：

# 加载保存的参数
# with open('libero_cameras_from_sim.json') as f:
#     params = json.load(f)

# 固定相机 - 直接使用
# K = params['agentview']['K']
# T = params['agentview']['T_world_to_cam']  # 固定值

# 手腕相机 - K固定，T每帧获取
# K_wrist = params['robot0_eye_in_hand']['K']
# T_wrist = get_camera_extrinsic_matrix(env.sim, "robot0_eye_in_hand")  # 每帧调用！
# T_wrist_inv = np.linalg.inv(T_wrist)
# ----------------------------------------------------------------------------------------------------
