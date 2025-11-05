import os, time, csv
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None
LOG_DIR = os.path.join(os.getcwd(), "runs")
os.makedirs(LOG_DIR, exist_ok=True)
CSV_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(CSV_DIR, exist_ok=True)
CSV_PATH = os.path.join(CSV_DIR, f"training_log_{int(time.time())}.csv")
PLOT_DIR = os.path.join(os.getcwd(), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
episodic_rewards = []
episode_lengths = []
episodic_success = []
episodic_strat_loss = []
episodic_tact_loss = []
epsilons = []
_current_episode_reward = 0.0
_current_episode_steps = 0
_current_episode_strat_losses = []
_current_episode_tact_losses = []
_current_episode_success = 0

with open(CSV_PATH, "w", newline="") as f:
    writer_csv = csv.writer(f)
    writer_csv.writerow([
        "episode", "total_reward", "length", "success",
        "avg_strat_loss", "avg_tact_loss", "epsilon", "timestamp"
    ])
_REWARD_WINDOW = 100
_reward_deque = deque(maxlen=_REWARD_WINDOW)
_success_deque = deque(maxlen=_REWARD_WINDOW)
def make_writer(exp_name: str | None = None):
    tag = exp_name or f"run_{int(time.time())}"
    logdir = os.path.join(LOG_DIR, tag)
    os.makedirs(logdir, exist_ok=True)
    if SummaryWriter is not None:
        try:
            w = SummaryWriter(log_dir=logdir)
            print(f"[logging_utils] SummaryWriter created at: {logdir}")
            return w
        except Exception as e:
            print(f"[logging_utils] Failed to create SummaryWriter at {logdir}: {e}")
    print("[logging_utils] TensorBoard not available; continuing with CSV/PNG logging")
    return None

def log_step(reward: float,
             action=None,
             env_info: dict | None = None,
             strat_loss: float | None = None,
             tact_loss: float | None = None,
             print_every_n_steps: int = 0):
    global _current_episode_reward, _current_episode_steps
    global _current_episode_strat_losses, _current_episode_tact_losses, _current_episode_success
    _current_episode_reward += float(reward)
    _current_episode_steps += 1
    if strat_loss is not None:
        _current_episode_strat_losses.append(float(strat_loss))
    if tact_loss is not None:
        _current_episode_tact_losses.append(float(tact_loss))
    if env_info is not None and env_info.get("success", False):
        _current_episode_success = 1
    if print_every_n_steps and _current_episode_steps % print_every_n_steps == 0:
        short = {
            "step": _current_episode_steps,
            "r_step": reward,
            "total_r": _current_episode_reward,
        }
        if env_info:
            for k in list(env_info)[:3]:
                short[k] = env_info.get(k)
        print("STEP LOG:", short)
def end_episode(episode_idx: int, epsilon: float, writer=None, save_plots: bool = False):
    global _current_episode_reward, _current_episode_steps, _current_episode_success
    global _current_episode_strat_losses, _current_episode_tact_losses
    global episodic_rewards, episode_lengths, episodic_success
    global episodic_strat_loss, episodic_tact_loss, epsilons
    avg_strat = float(np.mean(_current_episode_strat_losses)) if _current_episode_strat_losses else 0.0
    avg_tact = float(np.mean(_current_episode_tact_losses)) if _current_episode_tact_losses else 0.0
    episodic_rewards.append(_current_episode_reward)
    episode_lengths.append(_current_episode_steps)
    episodic_success.append(_current_episode_success)
    episodic_strat_loss.append(avg_strat)
    episodic_tact_loss.append(avg_tact)
    epsilons.append(epsilon)
    _reward_deque.append(_current_episode_reward)
    _success_deque.append(_current_episode_success)
    with open(CSV_PATH, "a", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow([
            episode_idx,
            _current_episode_reward,
            _current_episode_steps,
            _current_episode_success,
            avg_strat,
            avg_tact,
            epsilon,
            int(time.time())
        ])
    if writer is not None:
        try:
            writer.add_scalar("episode/total_reward", _current_episode_reward, episode_idx)
            writer.add_scalar("episode/length", _current_episode_steps, episode_idx)
            writer.add_scalar("episode/success", _current_episode_success, episode_idx)
            writer.add_scalar("loss/strategy", avg_strat, episode_idx)
            writer.add_scalar("loss/tactical", avg_tact, episode_idx)
            writer.add_scalar("policy/epsilon", epsilon, episode_idx)
            if len(_reward_deque) >= 1:
                writer.add_scalar("episode/reward_ma", float(np.mean(_reward_deque)), episode_idx)
                writer.add_scalar("episode/success_ma", float(np.mean(_success_deque)), episode_idx)
            writer.flush()
            print(f"[logging_utils] WROTE TB scalars for episode {episode_idx} (reward={_current_episode_reward})")
        except Exception as e:
            print(f"[logging_utils] Failed to write to writer for episode {episode_idx}: {e}")

    if save_plots and (episode_idx % 50 == 0):
        try:
            _save_plots_png(episode_idx)
        except Exception as e:
            print("[logging_utils] Failed to save PNG plots:", e)

    _current_episode_reward = 0.0
    _current_episode_steps = 0
    _current_episode_strat_losses = []
    _current_episode_tact_losses = []
    _current_episode_success = 0

def _save_plots_png(episode_idx: int):
    rewards = np.array(episodic_rewards)
    if rewards.size == 0:
        return
    win = min(100, max(10, int(len(rewards) / 10)))
    ma = np.convolve(rewards, np.ones(win)/win, mode='valid')
    plt.figure(figsize=(8,3))
    plt.plot(rewards, alpha=0.25)
    plt.plot(range(win-1, win-1+len(ma)), ma, linewidth=2)
    plt.title(f"Episode Reward (MA {win}) up to ep {episode_idx}")
    plt.xlabel('Episode'); plt.ylabel('Total Reward')
    plt.grid(True)
    plt.tight_layout()
    fn = os.path.join(PLOT_DIR, f"reward_ep_{episode_idx}.png")
    plt.savefig(fn, dpi=200)
    plt.close()
    success = np.array(episodic_success)
    if success.size > 0:
        ma_s = np.convolve(success, np.ones(win)/win, mode='valid')
        plt.figure(figsize=(6,2))
        plt.plot(range(win-1, win-1+len(ma_s)), ma_s, linewidth=2)
        plt.ylim(0,1)
        plt.title('Success rate (moving avg)')
        plt.xlabel('Episode'); plt.ylabel('Success')
        plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"success_ep_{episode_idx}.png"), dpi=200)
        plt.close()

    if len(episodic_strat_loss) > 0:
        plt.figure(figsize=(8,3))
        plt.plot(episodic_strat_loss, label='strat')
        plt.plot(episodic_tact_loss, label='tact')
        plt.title('Per-episode losses')
        plt.xlabel('Episode'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
        plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, f"losses_ep_{episode_idx}.png"), dpi=200); plt.close()

def moving_average(arr, window=100):
    arr = list(arr)
    if len(arr) < window:
        return np.array(arr)
    return np.convolve(np.array(arr), np.ones(window)/window, mode='valid')

def finalize_plots():
    try:
        _save_plots_png(len(episodic_rewards))
        print(f"[logging_utils] Saved final plots to {PLOT_DIR}")
    except Exception as e:
        print("[logging_utils] finalize_plots failed:", e)
