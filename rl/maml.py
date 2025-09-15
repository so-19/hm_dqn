import os, sys, torch
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy
import numpy as np
import torch
from .agents import DQNAgent
from .networks import StrategicDQN, TacticalDQN
from ..config import META
from drone import Drone
from uav_hm_dqn.envs.pybullet_envs import AVOIDANCE_SET
def inner_loop(env, strat_agent: DQNAgent, tact_agent: DQNAgent, steps: int, batch_size: int, meta_cfg=META, task_id=0, meta_iter=0):
    obs = env.reset()
    ep_steps = 0
    episode_count = 0
    total_reward = 0.0
    episode_reward = 0.0
    strat_losses = []
    tact_losses = []
    H_in = strat_agent.q.body[0].in_features if hasattr(strat_agent.q.body[0], 'in_features') else 12
    while ep_steps < steps:
        g = obs[:H_in]
        l = obs[H_in:H_in+14]
        aH = strat_agent.act(g, exploit=False)
        aL = tact_agent.act(l, exploit=False)
        act_final = aH
        obstacle_present = bool(l[0] > 0.5)
        if obstacle_present:
            name = Drone.ACTIONS[aL]
            if name in AVOIDANCE_SET:
                act_final = aL
        step = env.step(act_final)
        g2 = step.obs[:H_in]; l2 = step.obs[H_in:H_in+14]
        strat_agent.push(g, aH, step.reward, g2, step.done)
        tact_agent.push(l, aL, step.reward, l2, step.done)
        strat_loss = strat_agent.update(batch_size)
        tact_loss = tact_agent.update(batch_size)
        if strat_loss > 0:
            strat_losses.append(strat_loss)
        if tact_loss > 0:
            tact_losses.append(tact_loss)
        episode_reward += step.reward
        total_reward += step.reward
        obs = step.obs; ep_steps += 1
        if step.done:
            episode_count += 1
            if episode_count % 5 == 0:
                avg_strat_loss = np.mean(strat_losses) if strat_losses else 0.0
                avg_tact_loss = np.mean(tact_losses) if tact_losses else 0.0
                print(f"  Task {task_id+1}/{meta_cfg.tasks_per_batch} | Episode {episode_count} | "
                      f"Reward: {episode_reward:.2f} | Avg Strat Loss: {avg_strat_loss:.4f} | "
                      f"Avg Tact Loss: {avg_tact_loss:.4f} | Epsilon: {strat_agent.eps:.3f}")
            obs = env.reset()
            episode_reward = 0.0
    avg_strat_loss = np.mean(strat_losses) if strat_losses else 0.0
    avg_tact_loss = np.mean(tact_losses) if tact_losses else 0.0
    avg_episode_reward = total_reward / max(episode_count, 1)
    print(f"  Task {task_id+1} completed: {episode_count} episodes, "
          f"Avg reward: {avg_episode_reward:.2f}, "
          f"Avg losses - Strat: {avg_strat_loss:.4f}, Tact: {avg_tact_loss:.4f}")
    return avg_strat_loss, avg_tact_loss, avg_episode_reward

def meta_train(task_maker, det_fn, obs_dims, meta_cfg=META, rl_cfg=None, seed=0):
    start_time = time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    H_in, L_in = obs_dims
    nA = 3

    print(f"\nStarting Meta-Learning Training")
    print(f"Configuration:")
    print(f"   - Meta iterations: {meta_cfg.meta_iters}")
    print(f"   - Tasks per batch: {meta_cfg.tasks_per_batch}")
    print(f"   - Inner steps: {meta_cfg.inner_steps}")
    print(f"   - Outer learning rate: {meta_cfg.outer_lr}")
    print(f"   - Inner learning rate: {meta_cfg.inner_lr}")
    print(f"   - Strategic input dim: {H_in}")
    print(f"   - Tactical input dim: {L_in}")
    print(f"   - Number of actions: {nA}")
    print(f"   - Total training steps per task: {meta_cfg.inner_steps * 300}")
    print("="*80)
    qH = StrategicDQN(H_in, nA)
    qL = TacticalDQN(L_in, nA)
    optH = torch.optim.Adam(qH.parameters(), lr=meta_cfg.outer_lr)
    optL = torch.optim.Adam(qL.parameters(), lr=meta_cfg.outer_lr)
    all_strat_losses = []
    all_tact_losses = []
    all_rewards = []
    for it in range(meta_cfg.meta_iters):
        iter_start_time = time.time()
        print(f"\n Meta-Iteration {it+1}/{meta_cfg.meta_iters}")
        print("-" * 60)
        batch_strat_losses = []
        batch_tact_losses = []
        batch_rewards = []
        for t in range(meta_cfg.tasks_per_batch):
            print(f"\n Task {t+1}/{meta_cfg.tasks_per_batch} (Meta-iter {it+1})")
            env = task_maker(seed + it*131 + t, det_fn=det_fn)
            fH = StrategicDQN(H_in, nA); fH.load_state_dict(qH.state_dict())
            fL = TacticalDQN(L_in, nA); fL.load_state_dict(qL.state_dict())
            aH = DQNAgent(fH, H_in, nA)
            aL = DQNAgent(fL, L_in, nA)
            avg_strat_loss, avg_tact_loss, avg_reward = inner_loop(env, aH, aL, steps=meta_cfg.inner_steps * 300, batch_size=64, meta_cfg=meta_cfg, task_id=t, meta_iter=it)
            batch_strat_losses.append(avg_strat_loss)
            batch_tact_losses.append(avg_tact_loss)
            batch_rewards.append(avg_reward)
            with torch.no_grad():
                for p, fp in zip(qH.parameters(), fH.parameters()):
                    p.grad = (p - fp).detach()
                for p, fp in zip(qL.parameters(), fL.parameters()):
                    p.grad = (p - fp).detach()
            optH.step(); optH.zero_grad(set_to_none=True)
            optL.step(); optL.zero_grad(set_to_none=True)
            env.close()
        batch_avg_strat_loss = np.mean(batch_strat_losses)
        batch_avg_tact_loss = np.mean(batch_tact_losses)
        batch_avg_reward = np.mean(batch_rewards)
        all_strat_losses.append(batch_avg_strat_loss)
        all_tact_losses.append(batch_avg_tact_loss)
        all_rewards.append(batch_avg_reward)
        iter_time = time.time() - iter_start_time
        print(f"\n Meta-Iteration {it+1} Summary:")
        print(f"   Strategic Loss: {batch_avg_strat_loss:.4f}")
        print(f"   Tactical Loss:  {batch_avg_tact_loss:.4f}")
        print(f"   Average Reward: {batch_avg_reward:.2f}")
        print(f"   Iteration Time: {iter_time:.1f}s")
        if (it+1) % 10 == 0:
            recent_strat_loss = np.mean(all_strat_losses[-10:])
            recent_tact_loss = np.mean(all_tact_losses[-10:])
            recent_reward = np.mean(all_rewards[-10:])
            elapsed_time = time.time() - start_time
            eta = (elapsed_time / (it+1)) * (meta_cfg.meta_iters - (it+1))
            print(f"\n Progress Update (Last 10 iterations):")
            print(f"   Avg Strategic Loss: {recent_strat_loss:.4f}")
            print(f"   Avg Tactical Loss:  {recent_tact_loss:.4f}")
            print(f"   Avg Reward:         {recent_reward:.2f}")
            print(f"   Progress: {it+1}/{meta_cfg.meta_iters} ({100*(it+1)/meta_cfg.meta_iters:.1f}%)")
            print(f"   Elapsed Time: {elapsed_time/60:.1f}min | ETA: {eta/60:.1f}min")

    total_time = time.time() - start_time
    print(f"\n Training Complete!")
    print(f" Final Results:")
    print(f"   Total Meta-Iterations: {meta_cfg.meta_iters}")
    print(f"   Final Strategic Loss: {all_strat_losses[-1]:.4f}")
    print(f"   Final Tactical Loss:  {all_tact_losses[-1]:.4f}")
    print(f"   Final Average Reward: {all_rewards[-1]:.2f}")
    print(f"   Best Reward Achieved: {max(all_rewards):.2f}")
    print(f"   Total Training Time: {total_time/60:.1f} minutes")
    print(f"   Average Time per Iteration: {total_time/meta_cfg.meta_iters:.1f} seconds")
    print("="*80)
    return qH.state_dict(), qL.state_dict()
