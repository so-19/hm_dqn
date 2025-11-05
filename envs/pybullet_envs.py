import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pybullet as p
import pybullet_data
import numpy as np
import time
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
#from uav_hm_dqn.config import ENV, REW
#from uav_hm_dqn.utils.occupancy import compute_local_features
from config import ENV, REW
from utils.occupancy import compute_local_features
try:
    #from uav_hm_dqn.drone import Drone
    from drone import Drone
except Exception:
    Drone = None
@dataclass
class StepReturn:
    obs: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any]
AVOIDANCE_SET = {"yaw_left_S","yaw_right_S","strafe_left_S","strafe_right_S","brake","hover","maintain_position"}

class UAVWorld:
    def __init__(self, det_fn, seed: Optional[int]=None, cfg=ENV):
        self.det_fn = det_fn
        self.cfg = cfg
        self.client = None
        self.step_count = 0
        self.rng = np.random.default_rng(seed)
        self.goal = None
        self._drone_body = None
        self._drone = None
        self._last_dist = None
        self._last_action: Optional[int] = None
        self._last_vel = np.zeros(3, dtype=np.float32)

    def reset(self):
        if self.client is None:
            self.client = p.connect(p.GUI if self.cfg.gui else p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)
            p.setTimeStep(1.0 / self.cfg.hz)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        _ = p.loadURDF("plane.urdf")
        self._spawn_obstacles()
        self._spawn_drone()
        self.goal = self._rand_point_xy(z=1.0)
        self._spawn_goal_visual(self.goal)
        self.step_count = 0
        self._last_action = None
        self._last_dist = self._goal_dist()
        self._last_vel = np.zeros(3, dtype=np.float32)
        return self._get_obs()

    def close(self):
        if self.client is not None:
            p.disconnect(self.client)
            self.client = None
    def step(self, action: int) -> StepReturn:
        self._apply_action(action)
        for _ in range(4):
            p.stepSimulation()
            if self.cfg.gui:
                time.sleep(0.0)
        self.step_count += 1
        obs = self._get_obs()
        reward, done, info = self._compute_reward_done(action)
        return StepReturn(obs, reward, done, info)
    def _spawn_obstacles(self):
        for _ in range(self.cfg.obstacle_count):
            x, y = self._rand_xy()
            h = self.rng.uniform(0.6, 2.2)
            s = self.rng.uniform(self.cfg.obstacle_min, self.cfg.obstacle_max)
            if self.rng.random() < 0.55:
                col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[s, s, h/2])
                vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[s, s, h/2], rgbaColor=[0.4,0.4,0.4,1])
            else:
                col = p.createCollisionShape(p.GEOM_CYLINDER, radius=s, height=h)
                vis = p.createVisualShape(p.GEOM_CYLINDER, radius=s, length=h, rgbaColor=[0.4,0.4,0.4,1])
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                              basePosition=[x, y, h/2])
    def _spawn_drone(self):
        init_pos = [*self._rand_xy(), 1.0]
        init_orn = p.getQuaternionFromEuler([0,0,self.rng.uniform(-np.pi, np.pi)])
        if self.cfg.use_drone_py and Drone is not None:
            self._drone = Drone(init_pos, init_orn, urdf_path=self.cfg.drone_urdf)
            self._drone_body = self._drone.body
        else:
            self._drone = None
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.15,0.15,0.05])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.15,0.15,0.05], rgbaColor=[0,0,1,1])
            self._drone_body = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=col,baseVisualShapeIndex=vis, basePosition=init_pos,baseOrientation=init_orn)

    def _spawn_goal_visual(self, pos):
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.18, rgbaColor=[0,1,0,1])
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=pos)

    def _apply_action(self, a: int):
        if self._drone is not None:
            self._drone.apply_action(a)
            self._last_action = a
        else:
            pos, orn = p.getBasePositionAndOrientation(self._drone_body)
            yaw = p.getEulerFromQuaternion(orn)[2]
            move_step = 0.25; yaw_step = 0.1
            if a == 0:
                dx, dy = move_step*np.cos(yaw), move_step*np.sin(yaw)
                target = [pos[0]+dx, pos[1]+dy, pos[2]]
                p.resetBasePositionAndOrientation(self._drone_body, target, orn)
            elif a == 1:
                yaw += yaw_step
                new_orn = p.getQuaternionFromEuler([0,0,yaw])
                p.resetBasePositionAndOrientation(self._drone_body, pos, new_orn)
            self._last_action = a

    def _goal_dist(self):
        pos, _ = p.getBasePositionAndOrientation(self._drone_body)
        return float(np.linalg.norm(np.array(self.goal) - np.array(pos)))

    def _forward_left_right_clearances(self, yaw: float, rays=9, spread=0.6, max_d=8.0):
        pos, _ = p.getBasePositionAndOrientation(self._drone_body)
        origins_f, ends_f = [], []
        for i in range(rays):
            ang = yaw + (i - (rays-1)/2) * spread / max(1, (rays-1))
            ex, ey = pos[0] + max_d*np.cos(ang), pos[1] + max_d*np.sin(ang)
            origins_f.append([pos[0], pos[1], pos[2]])
            ends_f.append([ex, ey, pos[2]])
        hits_f = p.rayTestBatch(origins_f, ends_f)
        d_f = [max_d] * rays
        for i, hit in enumerate(hits_f):
            if hit[0] != -1:
                hit_pos = hit[3]
                d_f[i] = float(np.linalg.norm(np.array(hit_pos) - np.array(origins_f[i])))
        nearest_forward = float(min(d_f))
        ang_l = yaw + 0.9
        ang_r = yaw - 0.9
        def cast(ang):
            ex, ey = pos[0] + max_d*np.cos(ang), pos[1] + max_d*np.sin(ang)
            hit = p.rayTest([pos[0],pos[1],pos[2]], [ex,ey,pos[2]])[0]
            if hit[0] != -1:
                return float(np.linalg.norm(np.array(hit[3]) - np.array([pos[0],pos[1],pos[2]])))
            return max_d
        nearest_left = cast(ang_l)
        nearest_right = cast(ang_r)
        return nearest_forward, nearest_left, nearest_right

    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self._drone_body)
        lin_vel, ang_vel = p.getBaseVelocity(self._drone_body)
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)
        gx, gy, gz = self.goal
        dx, dy, dz = gx - pos[0], gy - pos[1], gz - pos[2]
        dist = float(np.sqrt(dx*dx + dy*dy + dz*dz))
        speed = float(np.linalg.norm(lin_vel))
        yaw_to_goal = float(np.arctan2(dy, dx) - yaw)
        yaw_to_goal = (yaw_to_goal + np.pi) % (2*np.pi) - np.pi
        if self._drone is not None:
            img_rgb = self._drone.camera_rgb()
        else:
            img_rgb = self._render_camera(pos, yaw)
        obstacle_present = self.det_fn(img_rgb)
        nearest_f, nearest_l, nearest_r = self._forward_left_right_clearances(yaw)
        local_feats = compute_local_features(img_rgb)
        global_vec = np.array([dx,dy,dz, dist,lin_vel[0],lin_vel[1],lin_vel[2],roll,pitch,yaw,yaw_to_goal, speed], dtype=np.float32)
        alt_err = float(pos[2] - REW.alt_target)
        local_vec = np.concatenate([
            np.array([float(obstacle_present),
                      nearest_f, nearest_l, nearest_r,
                      alt_err,
                      float(ang_vel[2])], dtype=np.float32),
            local_feats.astype(np.float32)
        ])
        return np.concatenate([global_vec, local_vec]).astype(np.float32)

    def _render_camera(self, pos, yaw):
        view = p.computeViewMatrix(
            cameraEyePosition=[pos[0], pos[1], pos[2] + 0.1],
            cameraTargetPosition=[pos[0] + np.cos(yaw), pos[1] + np.sin(yaw), pos[2] + 0.1],
            cameraUpVector=[0,0,1]
        )
        proj = p.computeProjectionMatrixFOV(ENV.fov, ENV.camera_width/ENV.camera_height, ENV.near_val, ENV.far_val)
        w, h, rgb, _, _ = p.getCameraImage(ENV.camera_width, ENV.camera_height, view, proj)
        rgb = np.reshape(rgb, (h, w, 4))[:, :, :3]
        return rgb

    def _compute_reward_done(self, action_id: int) -> Tuple[float,bool,Dict[str,Any]]:
        pos, orn = p.getBasePositionAndOrientation(self._drone_body)
        lin_vel, ang_vel = p.getBaseVelocity(self._drone_body)
        yaw = p.getEulerFromQuaternion(orn)[2]
        d = self._goal_dist()
        progress = REW.w_potential * (self._last_dist - d)
        self._last_dist = d
        gx, gy, gz = self.goal
        v = np.array(lin_vel, dtype=np.float32)
        to_goal = np.array([gx - pos[0], gy - pos[1], gz - pos[2]], dtype=np.float32)
        u_to_goal = to_goal / (np.linalg.norm(to_goal) + 1e-6)
        yaw_to_goal = np.arctan2(u_to_goal[1], u_to_goal[0]) - yaw
        yaw_to_goal = (yaw_to_goal + np.pi) % (2*np.pi) - np.pi
        heading_align = REW.w_heading * float(np.cos(yaw_to_goal))
        vel_to_goal = REW.w_vel_to_goal * float(np.dot(v, u_to_goal))
        nearest_f, _, _ = self._forward_left_right_clearances(yaw)
        clearance_reward = REW.w_clearance * float(np.clip((nearest_f - REW.safe_clearance)/REW.safe_clearance, 0.0, 1.0))
        proximity_pen = -REW.p_proximity * float(np.clip((REW.safe_clearance - nearest_f)/REW.safe_clearance, 0.0, 1.0))
        jerk_pen = -REW.p_jerk * float(np.linalg.norm(v - self._last_vel))
        self._last_vel = v.copy()
        if self._last_action is None:
            smooth_pen = 0.0
        else:
            smooth_pen = -REW.p_smooth * (0.0 if action_id == self._last_action else 1.0)
        alt_err = float(pos[2] - REW.alt_target)
        alt_pen = -REW.p_alt * abs(alt_err)
        ang_rate_pen = -REW.p_angular_rate * float(np.linalg.norm(ang_vel))
        time_pen = -REW.p_time
        control_pen = -REW.p_control
        obstacle_present = self.det_fn(self._drone.camera_rgb() if self._drone is not None else self._render_camera(pos,yaw))
        forward_like = False
        try:
            if self._drone is not None and 0 <= action_id < len(self._drone.ACTIONS):
                name = self._drone.ACTIONS[action_id]
                forward_like = ("forward" in name) or (name == "advance_forward_L")
            else:
                forward_like = (action_id == 0)
        except Exception:
            forward_like = False
        bad_forward_pen = -REW.p_bad_forward_with_obst if (obstacle_present and forward_like and nearest_f < REW.safe_clearance) else 0.0
        collided = len(p.getContactPoints(bodyA=self._drone_body)) > 0
        collision_pen = -REW.p_collision if collided else 0.0
        done = False
        success_bonus = 0.0
        if d < self.cfg.goal_radius:
            success_bonus = REW.w_success
            done = True
        if self.step_count >= self.cfg.max_steps or collided:
            done = True
        reward = float(
            progress + heading_align + vel_to_goal + clearance_reward + success_bonus
            + time_pen + control_pen + smooth_pen + jerk_pen + alt_pen + ang_rate_pen
            + proximity_pen + collision_pen + bad_forward_pen
        )
        info = dict(dist=d, progress=progress, heading=heading_align, vgoal=vel_to_goal,clearance=clearance_reward, prox_pen=proximity_pen, smooth_pen=smooth_pen,jerk_pen=jerk_pen, alt_pen=alt_pen, ang_rate_pen=ang_rate_pen,bad_forward_pen=bad_forward_pen, collided=collided)
        return reward, done, info
    def _rand_xy(self):
        s = self.cfg.world_size
        return float(self.rng.uniform(-s, s)), float(self.rng.uniform(-s, s))

    def _rand_point_xy(self, z=1.0):
        x, y = self._rand_xy()
        return [x, y, z]

def make_task(seed=None, det_fn=lambda img: False, cfg=ENV):
    env = UAVWorld(det_fn=det_fn, seed=seed, cfg=cfg)
    _ = env.reset()
    return env
