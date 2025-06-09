import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 26,
    "axes.titlesize": 24,
    "axes.labelsize": 24,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 24,
})


def create_dirs(*dirs):
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)


def remove_outliers(errors):
    q1 = np.percentile(errors, 25, axis=0)
    q3 = np.percentile(errors, 75, axis=0)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return errors[np.all((errors >= lower_bound) & (errors <= upper_bound), axis=1)]


def plot_three_phase(idx, observations, actions, save_dir="plots", csv_base=""):
    state_dir = os.path.join(save_dir, "states")
    action_dir = os.path.join(save_dir, "actions")
    os.makedirs(state_dir, exist_ok=True)
    os.makedirs(action_dir, exist_ok=True)

    # Plot states
    plt.figure(figsize=(10, 7))
    plt.plot(observations, linewidth=4, label=["Id", "Iq", "Id_ref", "Iq_ref"])
    plt.xlabel("Step")
    plt.ylabel("State Value")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(state_dir, f"{csv_base}_state_{idx}.png"), bbox_inches="tight")
    plt.close()

    # Plot actions
    plt.figure(figsize=(10, 7))
    plt.plot(actions, linewidth=4, label=["Vd", "Vq"])
    plt.xlabel("Step")
    plt.ylabel("Action Value")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(action_dir, f"{csv_base}_action_{idx}.png"), bbox_inches="tight")
    plt.close()




def plot_torque_tracking(idx, observations, actions, save_dir="plots", csv_base=""):
    state_dir = os.path.join(save_dir, "torque_states")
    action_dir = os.path.join(save_dir, "torque_actions")
    os.makedirs(state_dir, exist_ok=True)
    os.makedirs(action_dir, exist_ok=True)

    # Plot torque states
    plt.figure(figsize=(10, 7))
    plt.plot(observations,linewidth=4, label=["Te", "Te_ref", "Id", "Iq"])
    plt.xlabel("Step")
    plt.ylabel("State Value")
    # plt.title(f"Torque Tracking States - Episode {idx}")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(state_dir, f"{csv_base}_torque_state_{idx}.png"), bbox_inches="tight")
    plt.close()

    # Plot actions
    plt.figure(figsize=(10, 7))
    plt.plot(actions, linewidth=4,label=["Vd", "Vq"])
    plt.xlabel("Step")
    plt.ylabel("Action Value")
    # plt.title(f"Torque Tracking Actions - Episode {idx}")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(action_dir, f"{csv_base}_torque_action_{idx}.png"), bbox_inches="tight")
    plt.close()



def plot_box(errors, save_dir="plots", csv_base=""):
    error_list = remove_outliers(errors)

    plt.figure(figsize=(10, 7))
    plt.boxplot(
        error_list,
        patch_artist=True,
        showmeans=True,
        showfliers=True,
        meanprops=dict(marker="D", markeredgecolor="black", markerfacecolor="white"),
        medianprops=dict(color="black"),
        labels=["Id error", "Iq error"]
    )
    plt.ylabel("Absolute Error")
    plt.title("Current Tracking Error Boxplot")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    filename = os.path.join(save_dir, f"{csv_base}_boxplot.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved boxplot to {filename}")


def plot_boxes(error_dict, save_dir="plots"):
    create_dirs(save_dir)
    colors = plt.cm.tab10.colors

    # --- Id error ---
    id_error_data = [remove_outliers(errors)[:, 0] for errors in error_dict.values()]
    fig_id, ax_id = plt.subplots(figsize=(10, 7))
    box1 = ax_id.boxplot(
        id_error_data,
        patch_artist=True,
        showmeans=True,
        showfliers=True,
        meanprops=dict(marker="D", markeredgecolor="black", markerfacecolor="white"),
        medianprops=dict(color="black"),
    )
    ax_id.set_ylabel("Id Error")
    # ax_id.set_title("Id Tracking Error")
    ax_id.set_xticks(range(1, len(error_dict) + 1))
    ax_id.set_xticklabels(error_dict.keys(), rotation=15)
    ax_id.grid(True, linestyle="--", alpha=0.4)
    for patch, color in zip(box1["boxes"], colors):
        patch.set_facecolor(color)
    plt.tight_layout()
    filename_id = os.path.join(save_dir, "id_error_boxplot.png")
    fig_id.savefig(filename_id, dpi=300)
    plt.show()
    print(f"Saved Id error boxplot to {filename_id}")

    # --- Iq error ---
    iq_error_data = [remove_outliers(errors)[:, 1] for errors in error_dict.values()]
    fig_iq, ax_iq = plt.subplots(figsize=(10, 7))
    box2 = ax_iq.boxplot(
        iq_error_data,
        patch_artist=True,
        showmeans=True,
        showfliers=True,
        meanprops=dict(marker="D", markeredgecolor="black", markerfacecolor="white"),
        medianprops=dict(color="black"),
    )
    ax_iq.set_ylabel("Iq Error")
    # ax_iq.set_title("Iq Tracking Error")
    ax_iq.set_xticks(range(1, len(error_dict) + 1))
    ax_iq.set_xticklabels(error_dict.keys(), rotation=15)
    ax_iq.grid(True, linestyle="--", alpha=0.4)
    for patch, color in zip(box2["boxes"], colors):
        patch.set_facecolor(color)
    plt.tight_layout()
    filename_iq = os.path.join(save_dir, "iq_error_boxplot.png")
    fig_iq.savefig(filename_iq, dpi=300)
    plt.show()
    print(f"Saved Iq error boxplot to {filename_iq}")


def plot_te_box(error_dict, save_dir="plots"):
    create_dirs(save_dir)
    colors = plt.cm.tab10.colors
    data = [remove_outliers(errors).ravel() for errors in error_dict.values()]

    plt.figure(figsize=(10, 7))
    bp = plt.boxplot(
        data,
        patch_artist=True,
        showmeans=True,
        showfliers=True,
        meanprops=dict(marker="D", markeredgecolor="black", markerfacecolor="white"),
        medianprops=dict(color="black")
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    plt.ylabel("Torque Error")
    # plt.title("Torque Tracking Error")
    plt.xticks(range(1, len(error_dict) + 1), error_dict.keys(), rotation=15)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    filename = os.path.join(save_dir, "te_error_boxplot.png")
    plt.savefig(filename, dpi=300)
    plt.show()
    print(f"Saved torque boxplot to {filename}")

def plot_reward(csv_files, model_labels, save_dir="plots", csv_base="reward"):
    import matplotlib.pyplot as plt
    import os

    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 7))

    for csv_file, label in zip(csv_files, model_labels):
        data = pd.read_csv(csv_file)
        steps = data.iloc[:, 0].values
        rewards = data.iloc[:, 1].values
        plt.plot(steps, rewards, label=label, linewidth=2)

    plt.xlabel("Step")
    plt.ylabel("Average Reward")
    # plt.title("Training Reward Curve")
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    filename = os.path.join(save_dir, f"{csv_base}_curve.png")
    plt.savefig(filename, dpi=300)
    plt.show()
    print(f"Saved reward curve to {filename}")
