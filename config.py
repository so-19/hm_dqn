from dataclasses import dataclass
import os
@dataclass
class EnvConfig:
    gui: bool = False
    hz: int = 60
    max_steps: int = 800
    world_size: float = 8.0
    obstacle_count: int = 30
    obstacle_min: float = 0.2
    obstacle_max: float = 0.8
    goal_radius: float = 0.35
    proximity_thresh: float = 1.2
    camera_width: int = 224
    camera_height: int = 224
    fov: float = 70.0
    near_val: float = 0.05
    far_val: float = 20.0
    drone_urdf: str | None = __import__("os").path.join(__import__("os").path.dirname(__file__), "assets", "drone.urdf")
    gui=True
    use_drone_py: bool = True                        # <— use Python drone wrapper

@dataclass
class RLConfig:
    gamma: float = 0.99
    lr: float = 3e-4
    batch_size: int = 128
    buffer_size: int = 200_000
    target_tau: float = 1.0
    target_update_every: int = 1000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 150_000
    grad_clip: float = 10.0
    double_dqn: bool = True 
    dueling: bool = True       
    n_step: int = 3           

@dataclass
class MetaConfig:
    inner_lr: float = 1e-4
    outer_lr: float = 5e-4
    inner_steps: int = 3
    tasks_per_batch: int = 6
    meta_iters: int = 600
    tactical_steps_per_waypoint: int = 30

@dataclass
class RewardConfig:
    w_potential: float = 1.6          # progress (potential-based)
    w_heading: float = 0.4            # yaw alignment toward goal
    w_vel_to_goal: float = 0.6        # velocity component toward goal
    w_clearance: float = 0.25         # reward for keeping safe forward clearance
    w_success: float = 15.0           # terminal success bonus

    # penalties
    p_time: float = 1e-3              # per-step time penalty
    p_control: float = 5e-4           # control usage penalty
    p_smooth: float = 0.02            # penalty for action changes
    p_jerk: float = 0.01              # penalty for |Δv| (jerk proxy)
    p_alt: float = 0.3                # altitude tracking error penalty
    p_angular_rate: float = 0.02      # penalize |ω|
    p_collision: float = 8.0          # collision penalty
    p_proximity: float = 0.8          # close-approach shaping penalty
    p_bad_forward_with_obst: float = 1.5  # forward-like action while obstacle flagged

    safe_clearance: float = 2.0 
    alt_target: float = 1.0   


@dataclass
class DetectorConfig:
    config_path: str | None = None
    weights_path: str | None = None
    obstacle_labels: tuple[str, ...] = (
        "person","car","truck","bus","bicycle","motorcycle",
        "tree","building","wall","pole","barrier","fence","bench","rock"
    )
    box_threshold: float = 0.30
    text_threshold: float = 0.25
    def resolve(self):
        import os, importlib
        if not self.config_path:
            try:
                gd = importlib.import_module("groundingdino")
                base = os.path.dirname(gd.__file__)
                cand = os.path.join(base, "config", "GroundingDINO_SwinT_OGC.py")
                if os.path.exists(cand):
                    self.config_path = cand
            except Exception:
                pass
        if not self.weights_path:
            try:
                here = os.path.dirname(__file__)
                cand = os.path.join(here, "assets", "groundingdino_swint_ogc.pth")
                if os.path.exists(cand):
                    self.weights_path = cand
            except Exception:
                pass

        return self
ENV = EnvConfig()
RL  = RLConfig()
META = MetaConfig()
REW = RewardConfig()
DET = DetectorConfig().resolve()
