import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uav_hm_dqn.detection.dino_detector import ObstacleDetector
from uav_hm_dqn.envs.pybullet_envs import make_task
from uav_hm_dqn.rl.maml import meta_train
from uav_hm_dqn.config import ENV, META
from uav_hm_dqn.drone import Drone
if __name__ == "__main__":
    print("Starting meta-training...")
    detector = ObstacleDetector()
    print("Detector initialized")
    def task_factory(seed, det_fn):
        return make_task(seed=seed, det_fn=det_fn, cfg=ENV)
    H_in = 12
    L_in = 14
    nA = len(Drone.ACTIONS)
    print(f"Input dimensions: H_in={H_in}, L_in={L_in}, nA={nA}")
    print("Starting meta-training...")
    sdH, sdL = meta_train(task_factory, det_fn=detector, obs_dims=(H_in, L_in), meta_cfg=META)
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(sdH, "checkpoints/strategic_meta.pth")
    torch.save(sdL, "checkpoints/tactical_meta.pth")
    print("Saved meta-parameters to checkpoints/")
