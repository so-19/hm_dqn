from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict, Iterable
import numpy as np
import pybullet as p
GRAV = 9.81
@dataclass
class DroneConfig:
    mass: float = 1.2
    body_xy: Tuple[float, float] = (0.30, 0.30)
    body_z: float = 0.08
    linear_damping: float = 0.04
    angular_damping: float = 0.02
    hover_z: float = 1.0
    kz: float = 18.0
    dz: float = 3.0    
    F_xy_max: float = 10.0
    F_z_max: float = 10.0
    Tau_roll_max: float = 0.6
    Tau_pitch_max: float = 0.6
    Tau_yaw_max: float = 0.5
    F_fwd_S: float = 4.0
    F_fwd_L: float = 8.0
    F_back_S: float = 3.0
    F_strafe_S: float = 3.0
    F_brake: float = 6.0
    Tau_yaw_S: float = 0.35
    Tau_pitch_S: float = 0.35
    Tau_roll_S: float = 0.35
    cam_w: int = 224
    cam_h: int = 224
    cam_fov: float = 70.0
    cam_near: float = 0.05
    cam_far: float = 20.0
    cam_height_offset: float = 0.10

class Drone:
    ACTIONS = [
        "advance_forward_L", "rotate_left_S", "maintain_position",
        "forward_S", "forward_L", "back_S",
        "strafe_left_S", "strafe_right_S",
        "ascend_S", "descend_S",
        "yaw_left_S", "yaw_right_S",
        "pitch_fwd_S", "pitch_back_S",
        "roll_left_S", "roll_right_S",
        "brake", "hover", "land"
    ]

    def __init__(
        self,
        init_pos=(0, 0, 1.0),
        init_orn=(0, 0, 0, 1),
        urdf_path: Optional[str] = None,
        cfg: DroneConfig = DroneConfig(),
    ):
        self.cfg = cfg
        self.body = None
        self.target_z = float(init_pos[2]) if init_pos else cfg.hover_z
        if urdf_path:
            self.body = p.loadURDF(urdf_path, init_pos, init_orn)
            self.mass = p.getDynamicsInfo(self.body, -1)[0]
        else:
            hx, hy, hz = self.cfg.body_xy[0] / 2, self.cfg.body_xy[1] / 2, self.cfg.body_z / 2
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, hz], rgbaColor=[0.0, 0.0, 1.0, 1.0])
            self.body = p.createMultiBody(
                baseMass=self.cfg.mass,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=init_pos,
                baseOrientation=init_orn,
            )
            self.mass = self.cfg.mass
        p.changeDynamics(self.body, -1, linearDamping=self.cfg.linear_damping, angularDamping=self.cfg.angular_damping)

    def get_state(self):
        pos, orn = p.getBasePositionAndOrientation(self.body)
        lin_vel, ang_vel = p.getBaseVelocity(self.body)
        return pos, orn, lin_vel, ang_vel

    def camera_rgb(self) -> np.ndarray:
        pos, orn = p.getBasePositionAndOrientation(self.body)
        yaw = p.getEulerFromQuaternion(orn)[2]
        eye = [pos[0], pos[1], pos[2] + self.cfg.cam_height_offset]
        target = [eye[0] + np.cos(yaw), eye[1] + np.sin(yaw), eye[2]]
        up = [0, 0, 1]
        view = p.computeViewMatrix(cameraEyePosition=eye, cameraTargetPosition=target, cameraUpVector=up)
        aspect = self.cfg.cam_w / float(self.cfg.cam_h)
        proj = p.computeProjectionMatrixFOV(self.cfg.cam_fov, aspect, self.cfg.cam_near, self.cfg.cam_far)
        w, h, rgba, _, _ = p.getCameraImage(self.cfg.cam_w, self.cfg.cam_h, view, proj)
        rgb = np.reshape(rgba, (h, w, 4))[:, :, :3].astype(np.uint8)
        return rgb

    def _apply_hover(self):
        pos, orn, lin_vel, _ = self.get_state()
        z_err = self.target_z - pos[2]
        F_hover = self.mass * GRAV
        Fz = F_hover + self.cfg.kz * z_err - self.cfg.dz * lin_vel[2]
        p.applyExternalForce(self.body, -1, [0.0, 0.0, Fz], pos, p.WORLD_FRAME)

    def _dir_unit_vectors(self, yaw: float):
        fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float32)
        left = np.array([-np.sin(yaw), np.cos(yaw), 0.0], dtype=np.float32)
        return fwd, left

    def nudge_altitude(self, dz: float):
        self.target_z = max(0.1, self.target_z + float(dz))

    def set_target_altitude(self, z: float):
        self.target_z = max(0.1, float(z))
    def apply_action(self, action: Union[int, str, Iterable[float], Dict[str, object]]):
        self._apply_hover()
        if isinstance(action, dict) and action.get("mode") == "continuous":
            u = np.asarray(action["u"], dtype=np.float32)
            return self._apply_continuous(u)
        if isinstance(action, (list, tuple, np.ndarray)) and len(action) == 6:
            u = np.asarray(action, dtype=np.float32)
            return self._apply_continuous(u)
        if isinstance(action, int):
            if action < 0 or action >= len(self.ACTIONS):
                raise ValueError(f"Discrete action {action} out of range (0..{len(self.ACTIONS)-1})")
            key = self.ACTIONS[action]
        elif isinstance(action, str):
            key = action
            if key not in self.ACTIONS:
                raise ValueError(f"Unknown action '{key}'. Valid: {self.ACTIONS}")
        else:
            raise TypeError("Action must be int/str for discrete or length-6 array/dict for continuous.")
        pos, orn, lin_vel, _ = self.get_state()
        yaw = p.getEulerFromQuaternion(orn)[2]
        fwd, left = self._dir_unit_vectors(yaw)
        def apply_force(vec_world):
            vec = np.clip(vec_world, -self.cfg.F_xy_max, self.cfg.F_xy_max)
            p.applyExternalForce(self.body, -1, vec.tolist(), pos, p.WORLD_FRAME)
        def apply_torque_body(txyz):
            tx, ty, tz = txyz
            tx = np.clip(tx, -self.cfg.Tau_roll_max, self.cfg.Tau_roll_max)
            ty = np.clip(ty, -self.cfg.Tau_pitch_max, self.cfg.Tau_pitch_max)
            tz = np.clip(tz, -self.cfg.Tau_yaw_max, self.cfg.Tau_yaw_max)
            p.applyExternalTorque(self.body, -1, [tx, ty, tz], p.LINK_FRAME)
        if key == "advance_forward_L":
            apply_force(fwd * self.cfg.F_fwd_L)
            return
        if key == "rotate_left_S":
            apply_torque_body([0.0, 0.0, self.cfg.Tau_yaw_S])
            return
        if key == "maintain_position" or key == "hover":
            return
        if key == "forward_S":
            apply_force(fwd * self.cfg.F_fwd_S)
        elif key == "forward_L":
            apply_force(fwd * self.cfg.F_fwd_L)
        elif key == "back_S":
            apply_force(-fwd * self.cfg.F_back_S)
        elif key == "strafe_left_S":
            apply_force(left * self.cfg.F_strafe_S)
        elif key == "strafe_right_S":
            apply_force(-left * self.cfg.F_strafe_S)
        elif key == "ascend_S":
            self.nudge_altitude(+0.05)
        elif key == "descend_S":
            self.nudge_altitude(-0.05)
        elif key == "yaw_left_S":
            apply_torque_body([0.0, 0.0, +self.cfg.Tau_yaw_S])
        elif key == "yaw_right_S":
            apply_torque_body([0.0, 0.0, -self.cfg.Tau_yaw_S])
        elif key == "pitch_fwd_S":
            apply_torque_body([0.0, +self.cfg.Tau_pitch_S, 0.0])
        elif key == "pitch_back_S":
            apply_torque_body([0.0, -self.cfg.Tau_pitch_S, 0.0])
        elif key == "roll_left_S":
            apply_torque_body([+self.cfg.Tau_roll_S, 0.0, 0.0])
        elif key == "roll_right_S":
            apply_torque_body([-self.cfg.Tau_roll_S, 0.0, 0.0])
        elif key == "brake":
            vxy = np.array(lin_vel[:2], dtype=np.float32)
            if np.linalg.norm(vxy) > 1e-6:
                dir_opp = -vxy / (np.linalg.norm(vxy) + 1e-6)
                apply_force(np.array([dir_opp[0], dir_opp[1], 0.0]) * self.cfg.F_brake)
        elif key == "land":
            self.target_z = max(0.05, self.target_z - 0.02)
        else:
            raise RuntimeError(f"Unhandled action key '{key}'")

    def _apply_continuous(self, u: np.ndarray):
        if u.shape != (6,):
            raise ValueError("Continuous action must be length-6 array.")
        Fx_b = float(u[0]) * self.cfg.F_xy_max
        Fy_b = float(u[1]) * self.cfg.F_xy_max
        Fz_w = float(u[2]) * self.cfg.F_z_max
        Tx_b = float(u[3]) * self.cfg.Tau_roll_max
        Ty_b = float(u[4]) * self.cfg.Tau_pitch_max
        Tz_b = float(u[5]) * self.cfg.Tau_yaw_max
        pos, orn, _, _ = self.get_state()
        yaw = p.getEulerFromQuaternion(orn)[2]
        c, s = np.cos(yaw), np.sin(yaw)
        Fx_w = c * Fx_b - s * Fy_b
        Fy_w = s * Fx_b + c * Fy_b
        p.applyExternalForce(self.body, -1, [Fx_w, Fy_w, Fz_w], pos, p.WORLD_FRAME)
        p.applyExternalTorque(self.body, -1, [Tx_b, Ty_b, Tz_b], p.LINK_FRAME)