from dataclasses import dataclass
import os
@dataclass
class EnvConfig:
    gui: bool = False                    
    hz: int = 60           
    max_steps: int = 1200                
    world_size: float = 10.0    
    obstacle_count: int = 25
    obstacle_min: float = 0.2
    obstacle_max: float = 0.8
    goal_radius: float = 0.35
    proximity_thresh: float = 1.5
    camera_width: int = 128               
    camera_height: int = 128
    fov: float = 70.0
    near_val: float = 0.05
    far_val: float = 20.0
    drone_urdf: str | None = os.path.join(
        os.path.dirname(__file__), "assets", "drone.urdf"
    )
    gui = False                          
    use_drone_py: bool = True

@dataclass
class RLConfig:
    gamma: float = 0.995                 
    lr: float = 2e-4                    
    batch_size: int = 128
    buffer_size: int = 500_000
    target_tau: float = 0.005
    target_update_every: int = 500
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
    meta_iters: int = 800 
    tactical_steps_per_waypoint: int = 30

@dataclass
class RewardConfig:
    w_potential: float = 1.6         
    w_heading: float = 0.4          
    w_vel_to_goal: float = 0.6        
    w_clearance: float = 0.25       
    w_success: float = 15.0      
    p_time: float = 1e-3        
    p_control: float = 5e-4      
    p_smooth: float = 0.02       
    p_jerk: float = 0.01            
    p_alt: float = 0.25             
    p_angular_rate: float = 0.02    
    p_collision: float = 8.0
    p_proximity: float = 0.8         
    p_bad_forward_with_obst: float = 1.2
    safe_clearance: float = 2.0
    alt_target: float = 1.0
    use_shaping: bool = True
    w_alt_shaping: float = 0.3         
    w_progress_shaping: float = 0.8      
    r_alive: float = 0.02                

@dataclass
class DetectorConfig:
    config_path: str | None = None
    weights_path: str | None = None
    obstacle_labels: tuple[str, ...] = (
        "person", "car", "truck", "bus", "bicycle", "motorcycle",
        "tree", "building", "wall", "pole", "barrier", "fence", "bench", "rock"
    )
    box_threshold: float = 0.30
    text_threshold: float = 0.25
    def resolve(self):
        import importlib
        try:
            if not self.config_path:
                gd = importlib.import_module("groundingdino")
                base = os.path.dirname(gd.__file__)
                cand = os.path.join(base, "config", "GroundingDINO_SwinT_OGC.py")
                if os.path.exists(cand):
                    self.config_path = cand
            if not self.weights_path:
                here = os.path.dirname(__file__)
                cand = os.path.join(here, "assets", "groundingdino_swint_ogc.pth")
                if os.path.exists(cand):
                    self.weights_path = cand
        except Exception:
            pass
        return self
ENV = EnvConfig()
RL = RLConfig()
META = MetaConfig()
REW = RewardConfig()
DET = DetectorConfig().resolve()
