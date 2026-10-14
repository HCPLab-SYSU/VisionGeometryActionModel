import numpy as np
from robosuite.utils.observables import Observable
import robosuite.utils.transform_utils as T
from scipy.spatial.transform import Rotation as R


def _image_key(camera_name: str) -> str:
    return f"{camera_name}_image"


def _depth_key(camera_name: str) -> str:
    return f"{camera_name}_depth"


def _intrinsic_key(camera_name: str) -> str:
    return f"{camera_name}_intrinsic"


def _extrinsic_key(camera_name: str) -> str:
    return f"{camera_name}_extrinsic"


class LiberoCameraAdapter:
    """
    Lightweight camera adapter for LIBERO envs.
    It wraps an existing env and adds:
    - camera pose variations
    - intrinsic / extrinsic outputs
    - optional depth conversion
    - optional HD rendering observable
    """

    def __init__(
        self,
        base_env,
        img_height=256,
        img_width=256,
        cameras=("agentview", "robot0_eye_in_hand"),
        camera_pose_variations=False,
        hd_rendering=False,
        hd_size=512,  # 默认512*512
        enable_depth=False,
    ):
        self.base_env = base_env
        self.img_height = img_height
        self.img_width = img_width
        self.cameras = tuple(cameras)
        self.camera_pose_variations = camera_pose_variations
        self.hd_rendering = hd_rendering  # 开启了hd_rendering之后，obs里面依然是原始分辨率的图片，高分辨率的图片会被存到env.render_out里面，或者通过env.render()函数获取
        self.hd_size = hd_size
        self.enable_depth = enable_depth
        self.render_out = None

        self.cam_id = None
        self.new_position = None
        self.new_rotation = None

        if self.hd_rendering:
            self._maybe_register_hd_observable()

        if self.camera_pose_variations:
            self._setup_camera_pose_variation()

    @property
    def env(self):
        return self.base_env

    @property
    def sim(self):
        return self.base_env.sim

    def close(self):
        return self.base_env.close()

    def reset(self):
        raw_obs = self.base_env.reset()
        if self.camera_pose_variations:
            self._apply_camera_pose_variation()
            raw_obs = self._refresh_observation()
        return self.make_obs(raw_obs)

    def set_init_state(self, init_state):
        raw_obs = self.base_env.set_init_state(init_state)
        if self.camera_pose_variations:
            self._apply_camera_pose_variation()
            raw_obs = self._refresh_observation()
        return self.make_obs(raw_obs)

    def get_observation(self):
        if hasattr(self.base_env, "get_observation"):
            raw_obs = self.base_env.get_observation()
        else:
            raw_obs = self._refresh_observation()
        return self.make_obs(raw_obs)

    def step(self, action):
        raw_obs, reward, done, info = self.base_env.step(action)
        obs = self.make_obs(raw_obs)
        return obs, reward, done, info

    def render(self, *args, **kwargs):
        if self.render_out is not None:
            return self.render_out
        if hasattr(self.base_env, "render"):
            return self.base_env.render(*args, **kwargs)
        return None

    def _inner_env(self):
        return getattr(self.base_env, "env", self.base_env)

    def _refresh_observation(self):
        if hasattr(self.base_env, "regenerate_obs_from_state") and hasattr(self.base_env, "get_sim_state"):
            return self.base_env.regenerate_obs_from_state(self.base_env.get_sim_state().flatten())
        if hasattr(self.base_env, "env") and hasattr(self.base_env.env, "_get_observations"):
            return self.base_env.env._get_observations()
        raise AttributeError("Cannot refresh observation from wrapped env.")

    def _maybe_register_hd_observable(self):
        inner = self._inner_env()
        if not hasattr(inner, "_create_camera_sensors") or not hasattr(inner, "add_observable"):
            return
        new_sensors, _ = inner._create_camera_sensors(
            cam_name="agentview",
            cam_w=self.hd_size,
            cam_h=self.hd_size,
            cam_d=False,
            cam_segs=None,
        )
        observable = Observable(
            name="agentview_image_hd",
            sensor=new_sensors[0],
            sampling_rate=inner.control_freq,
        )
        inner.add_observable(observable)

    def _setup_camera_pose_variation(self):
        camera_name = "agentview"
        self.cam_id = self.sim.model.camera_name2id(camera_name)
        old_position = self.sim.model.cam_pos[self.cam_id].copy()
        old_rotation = self.sim.model.cam_quat[self.cam_id].copy()

        variation = self.camera_pose_variations
        if isinstance(variation, str):
            if variation == "small":
                self.new_position = old_position + np.array([0.0, 0.3, -0.1])
                self.new_rotation = np.array([0.44834694, 0.2579209, 0.37187116, 0.77082661])
            elif variation == "medium":
                self.new_position = old_position + np.array([-0.2, 0.7, -0.2])
                self.new_rotation = np.array([0.16658396, 0.23584841, 0.47143382, 0.83329194])
            elif variation == "large":
                self.new_position = old_position + np.array([-1.2, 1.0, -0.2])
                self.new_rotation = np.array([-0.14345217, -0.20207276, 0.57364103, 0.78072021])
            else:
                raise ValueError(f"invalid camera_pose_variation: {variation}")
        elif isinstance(variation, (int, float)):
            base_pos = self.sim.data.body("gripper0_eef").xpos
            theta = float(variation)
            r = old_position[0] - base_pos[0]
            x = r * np.cos(theta) + base_pos[0]
            y = r * np.sin(theta) + base_pos[1]
            z = old_position[2]
            self.new_position = np.array([x, y, z])

            r_z = R.from_euler("z", theta).as_matrix()
            r_ref = R.from_quat(old_rotation, scalar_first=True).as_matrix()
            r_total = r_z @ r_ref
            self.new_rotation = R.from_matrix(r_total).as_quat(scalar_first=True)
        else:
            raise ValueError("camera_pose_variations must be False, str, int or float")

    def _apply_camera_pose_variation(self):
        if self.cam_id is None:
            return
        self.sim.model.cam_pos[self.cam_id] = self.new_position
        self.sim.model.cam_quat[self.cam_id] = self.new_rotation
        self.sim.forward()

    def make_obs(self, raw_obs):
        """
        把旧的raw_obs重新编辑并保存到新的obs中，特别是处理内外参矩阵以及深度图
        """
        obs = {}

        if self.hd_rendering and "agentview_image_hd" in raw_obs:
            self.render_out = raw_obs["agentview_image_hd"]
        elif _image_key(self.cameras[0]) in raw_obs:
            self.render_out = raw_obs[_image_key(self.cameras[0])]

        # Keep all non-image keys by default for compatibility.
        for key, value in raw_obs.items():
            if key.endswith("_image") or key.endswith("_depth") or key == "agentview_image_hd":
                continue
            obs[key] = value

        for cam_name in self.cameras:
            img_key = _image_key(cam_name)
            if img_key in raw_obs:
                obs[img_key] = np.ascontiguousarray(raw_obs[img_key])

            dep_key = _depth_key(cam_name)
            if self.enable_depth and dep_key in raw_obs:
                depth_raw = raw_obs[dep_key].squeeze()
                depth_meters = self.depth_im_to_meters(depth_raw)
                depth = np.ascontiguousarray((depth_meters * 1000).astype(np.uint16))
                obs[dep_key] = depth[..., np.newaxis]
            elif dep_key in raw_obs:
                obs[dep_key] = np.ascontiguousarray(raw_obs[dep_key])

            obs[_intrinsic_key(cam_name)] = self.get_camera_intrinsic_matrix(cam_name)
            obs[_extrinsic_key(cam_name)] = self.get_camera_extrinsic_matrix(cam_name)

        return obs

    def depth_im_to_meters(self, depth_im):
        extent = self.sim.model.stat.extent
        near = self.sim.model.vis.map.znear * extent
        far = self.sim.model.vis.map.zfar * extent
        depth_im_m = near / (1 - np.array(depth_im) * (1 - near / far))
        return depth_im_m

    def get_camera_intrinsic_matrix(self, camera_name):
        cam_id = self.sim.model.camera_name2id(camera_name)
        fovy = self.sim.model.cam_fovy[cam_id]
        f = 0.5 * self.img_height / np.tan(fovy * np.pi / 360.0)
        return np.array(
            [[f, 0.0, self.img_width / 2.0], [0.0, f, self.img_height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    def get_camera_extrinsic_matrix(self, camera_name):
        cam_id = self.sim.model.camera_name2id(camera_name)
        camera_pos = self.sim.data.cam_xpos[cam_id]
        camera_rot = self.sim.data.cam_xmat[cam_id].reshape(3, 3)
        pose = T.make_pose(camera_pos, camera_rot)
        camera_axis_correction = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        return (pose @ camera_axis_correction).astype(np.float32)
