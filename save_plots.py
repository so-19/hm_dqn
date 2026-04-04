# This script extracts scalar plots from TensorBoard logs and saves them as PNG images.

import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# 🔹 Path to TensorBoard logs
log_dir = "runs/uav_hm_dqn_experiment1"

# 🔹 Output folder
output_dir = "plots/Converted_plots"
os.makedirs(output_dir, exist_ok=True)

# 🔹 Load event file
ea = event_accumulator.EventAccumulator(log_dir)
ea.Reload()

# 🔹 Get all scalar tags (graphs)
tags = ea.Tags()["scalars"]

print("Found scalar plots:", tags)

# 🔹 Loop through each graph
for tag in tags:
    events = ea.Scalars(tag)

    steps = [e.step for e in events]
    values = [e.value for e in events]

    plt.figure()
    plt.plot(steps, values)
    plt.xlabel("Step")
    plt.ylabel(tag)
    plt.title(tag)

    # Clean filename
    filename = tag.replace("/", "_") + ".png"
    filepath = os.path.join(output_dir, filename)

    plt.savefig(filepath)
    plt.close()

    print(f"Saved: {filepath}")

print("✅ All TensorBoard plots saved!")