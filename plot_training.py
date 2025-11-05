import os
import numpy as np
import matplotlib.pyplot as plt
from logging_utils import moving_average, episodic_rewards, episodic_strat_loss, episodic_tact_loss, epsilons, episodic_success, CSV_PATH
def load_csv_if_needed():
    if episodic_rewards:
        return episodic_rewards, episodic_strat_loss, episodic_tact_loss, epsilons, episodic_success
    import csv
    rewards, strat, tact, eps, succ = [], [], [], [], []
    with open(CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rewards.append(float(row["total_reward"]))
            strat.append(float(row["avg_strat_loss"]))
            tact.append(float(row["avg_tact_loss"]))
            eps.append(float(row["epsilon"]))
            succ.append(int(row["success"]))
    return np.array(rewards), np.array(strat), np.array(tact), np.array(eps), np.array(succ)
rewards, strat_loss, tact_loss, eps, success = load_csv_if_needed()
win = 100
ma_rewards = moving_average(rewards, window=win)
ma_success = moving_average(success, window=win)
os.makedirs("plots", exist_ok=True)
plt.figure(figsize=(10,5))
plt.plot(rewards, alpha=0.2, label="episodic reward")
plt.plot(range(win-1, win-1+len(ma_rewards)), ma_rewards, label=f"{win}-episode MA", linewidth=2)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Episodic Reward and Moving Average")
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig("plots/reward_ma.png", dpi=200)
plt.close()
plt.figure(figsize=(10,4))
plt.plot(range(win-1, win-1+len(ma_success)), ma_success, label=f"{win}-ep success rate", linewidth=2)
plt.xlabel("Episode"); plt.ylabel("Success Rate"); plt.title("Success Rate (moving average)")
plt.ylim(0,1.0); plt.grid(True); plt.tight_layout()
plt.savefig("plots/success_rate_ma.png", dpi=200); plt.close()
plt.figure(figsize=(10,5))
plt.plot(strat_loss, label="strategy loss"); plt.plot(tact_loss, label="tactical loss")
plt.xlabel("Episode"); plt.ylabel("Loss"); plt.title("Per-episode losses"); plt.legend(); plt.grid(True)
plt.tight_layout(); plt.savefig("plots/losses.png", dpi=200); plt.close()
plt.figure(figsize=(8,3))
plt.plot(eps, label="epsilon"); plt.xlabel("Episode"); plt.ylabel("Epsilon"); plt.title("Epsilon Decay"); plt.grid(True)
plt.tight_layout(); plt.savefig("plots/epsilon.png", dpi=200); plt.close()
print("Saved plots to ./plots")
