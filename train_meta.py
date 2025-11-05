import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drone import Drone
from config import ENV, META
from detection.dino_detector import ObstacleDetector
from rl.maml import meta_train
from envs.pybullet_envs import make_task
from logging_utils import make_writer
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
    writer = make_writer("uav_hm_dqn_experiment1")
    if writer is None:
        print("[train_meta] WARNING: writer is None (no tensorboard available). Fall back to CSV logging only.")
    else:
        print("[train_meta] TensorBoard writer initialized.")

    print("Starting meta-training...")
    sdH, sdL = meta_train(task_factory, det_fn=detector, obs_dims=(H_in, L_in), meta_cfg=META, writer=writer)
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(sdH, "checkpoints/strategic_meta.pth")
    torch.save(sdL, "checkpoints/tactical_meta.pth")
    print("Saved meta-parameters to checkpoints/")

    if writer is not None:
        try:
            writer.close()
            print("[train_meta] Closed TensorBoard writer.")
        except Exception:
            pass
