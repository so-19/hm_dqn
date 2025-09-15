import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
from detection.dino_detector import ObstacleDetector
from uav_hm_dqn.envs.pybullet_envs import make_task, AVOIDANCE_SET
from rl.networks import StrategicDQN, TacticalDQN
from rl.agents import DQNAgent
from config import ENV, META, RL
from drone import Drone
def specialize_and_run(metaH_path, metaL_path, episodes_specialize=2):
    detector = ObstacleDetector()
    env = make_task(seed=None, det_fn=detector, cfg=ENV)
    H_in, L_in, nA = 12, 14, len(Drone.ACTIONS)
    qH = StrategicDQN(H_in, nA, dueling=RL.dueling)
    qL = TacticalDQN(L_in, nA, dueling=RL.dueling)
    qH.load_state_dict(torch.load(metaH_path, map_location="cpu"))
    qL.load_state_dict(torch.load(metaL_path, map_location="cpu"))
    aH = DQNAgent(qH, H_in, nA)
    aL = DQNAgent(qL, L_in, nA)
    def split_obs(obs):
        return obs[:H_in], obs[H_in:H_in+L_in]
    for _ in range(episodes_specialize):
        obs = env.reset(); done = False
        while not done:
            g, l = split_obs(obs)
            a_strat = aH.act(g, exploit=False)
            a_tact  = aL.act(l, exploit=False)
            obstacle_present = bool(l[0] > 0.5)
            act_final = a_strat
            if obstacle_present:
                name = Drone.ACTIONS[a_tact]
                if name in AVOIDANCE_SET:
                    act_final = a_tact
            step = env.step(act_final)
            g2, l2 = split_obs(step.obs)
            aH.push(g, a_strat, step.reward, g2, step.done)
            aL.push(l, a_tact, step.reward, l2, step.done)
            aH.update(RL.batch_size); aL.update(RL.batch_size)
            obs = step.obs; done = step.done
    aH.eps = 0.0; aL.eps = 0.0
    obs = env.reset(); done = False; total_r = 0.0; steps = 0
    while not done:
        g, l = split_obs(obs)
        a_strat = aH.act(g, exploit=True)
        a_tact  = aL.act(l, exploit=True)
        obstacle_present = bool(l[0] > 0.5)
        act_final = a_strat
        if obstacle_present:
            name = Drone.ACTIONS[a_tact]
            if name in AVOIDANCE_SET:
                act_final = a_tact
        step = env.step(act_final)
        obs = step.obs; done = step.done
        total_r += step.reward; steps += 1
    env.close()
    print(f"Episode return: {total_r:.2f} in {steps} steps")
if __name__ == "__main__":
    specialize_and_run("checkpoints/strategic_meta.pth", "checkpoints/tactical_meta.pth", episodes_specialize=3)