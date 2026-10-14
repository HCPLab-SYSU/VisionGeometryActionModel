#!/usr/bin/env python3
"""
Generate Gaussian heat map + depth map ground truth for LIBERO-10.
Output: .npz with heatmap [N,H,W], depthmap [N,H,W], ee_xyz [N,3], valid [N]
"""
import json
import numpy as np
import h5py
from pathlib import Path

# Config
CAMERA_PARAMS_PATH = "/home/lqc/dreamzero/scripts/data/libero/libero_10_cameras_params.json"
DATA_DIR = "/home/lqc/dreamzero/data/libero_10"  # adjust if needed
OUTPUT_PATH = "/home/lqc/dreamzero/data/libero_10_gt_heatmap_depth.npz"
IMG_SIZE = 256
LATENT_SIZE = 32  # 256/8 for VAE latent
SIGMA = 2.0  # Gaussian heat map sigma


def project_xyz_to_uvd(xyz, K, T_w2c):
    """xyz (N,3) world → uv (N,2), depth (N,), valid (N,)"""
    N = len(xyz)
    xyz_h = np.concatenate([xyz, np.ones((N, 1))], axis=1)
    c = (T_w2c @ xyz_h.T).T  # 变换到相机坐标系
    x, y, z = c[:, 0], c[:, 1], c[:, 2]  # z是深度
    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    valid = (z > 0) & (u >= 0) & (u < IMG_SIZE) & (v >= 0) & (v < IMG_SIZE)  # 判断(u,v)是否在图像内且z>0
    return np.stack([u, v], axis=1), z, valid


def generate_gaussian_heatmap(uv, size=LATENT_SIZE, sigma=SIGMA):
    """uv (2,) in [0,256] → heatmap (H,W) at latent size"""
    uv_latent = uv / 8.0  # scale to latent space
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    dist2 = (xx - uv_latent[0])**2 + (yy - uv_latent[1])**2
    heatmap = np.exp(-dist2 / (2 * sigma**2))
    return heatmap.astype(np.float32)


def generate_depth_map(depth, uv, size=LATENT_SIZE, sigma=SIGMA, depth_min=0.3, depth_max=2.0):
    """depth scalar + uv (2,) → normalized depth map (H,W)"""
    depth_norm = np.clip((depth - depth_min) / (depth_max - depth_min), 0, 1)
    uv_latent = uv / 8.0
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    weight = np.exp(-((xx - uv_latent[0])**2 + (yy - uv_latent[1])**2) / (2 * sigma**2))
    depthmap = depth_norm * weight
    return depthmap.astype(np.float32)


TASK_ORDER = [
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
]


def main():
    cam_params = json.load(open(CAMERA_PARAMS_PATH))
    heatmaps, depthmaps = [], []
    ee_xyz_list, valid_list = [], []
    ee_uv_list, ee_depth_list = [], []
    episode_ends = []
    prev_end = 0

    for task in TASK_ORDER:
        hdf5_path = Path(DATA_DIR) / f"{task}_demo.hdf5"
        if not hdf5_path.exists():
            print(f"Skip {task}: file not found")
            continue

        K = np.array(cam_params[task]["K"], dtype=np.float64)
        T_w2c = np.array(cam_params[task]["T_world_to_cam"], dtype=np.float64)

        with h5py.File(hdf5_path, "r") as f:
            demos = f["data"]
            for i in range(len(demos)):
                ee_pos = demos[f"demo_{i}"]["obs"]["ee_pos"][:]  # (T,3)
                uv, depth, valid = project_xyz_to_uvd(ee_pos, K, T_w2c)

                for t in range(len(ee_pos)):
                    if valid[t]:
                        hm = generate_gaussian_heatmap(uv[t])
                        dm = generate_depth_map(depth[t], uv[t])
                    else:
                        hm = np.zeros((LATENT_SIZE, LATENT_SIZE), dtype=np.float32)
                        dm = np.zeros((LATENT_SIZE, LATENT_SIZE), dtype=np.float32)
                    
                    heatmaps.append(hm)
                    depthmaps.append(dm)
                    ee_xyz_list.append(ee_pos[t])
                    valid_list.append(valid[t])
                    ee_uv_list.append(uv[t])
                    ee_depth_list.append(depth[t])

                prev_end += len(ee_pos)
                episode_ends.append(prev_end)

        print(f"✓ {task}")

    heatmaps = np.stack(heatmaps, axis=0)  # (N,H,W)
    depthmaps = np.stack(depthmaps, axis=0)
    ee_xyz = np.stack(ee_xyz_list, axis=0)
    ee_uv = np.stack(ee_uv_list, axis=0)
    ee_depth = np.stack(ee_depth_list, axis=0)
    valid = np.array(valid_list, dtype=bool)
    episode_ends = np.array(episode_ends, dtype=np.int64)

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_PATH,
        heatmaps=heatmaps,
        depthmaps=depthmaps,
        ee_xyz=ee_xyz,
        ee_uv=ee_uv,
        ee_depth=ee_depth,
        valid=valid,
        episode_ends=episode_ends,
    )

    print(f"\n✓ Generated {len(heatmaps)} frames")
    print(f"  Valid: {valid.sum()} ({100*valid.mean():.1f}%)")
    print(f"  Episodes: {len(episode_ends)}")
    print(f"  Heatmap shape: {heatmaps.shape}")
    print(f"  Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
