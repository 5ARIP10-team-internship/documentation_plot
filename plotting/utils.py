import os

import numpy as np
from matplotlib import pyplot as plt


def plot_three_phase(idx, observations, actions, save_dir="plots", csv_base=""):
    PLT_STATE_DIR = save_dir + "/states/"
    PLT_ACTION_DIR = save_dir + "/actions/"

    if not os.path.exists(PLT_STATE_DIR):
        os.makedirs(PLT_STATE_DIR)

    if not os.path.exists(PLT_ACTION_DIR):
        os.makedirs(PLT_ACTION_DIR)

    # Plot state
    plt.figure()
    plt.plot(observations, label=["Id", "Iq", "Idref", "Iqref"])
    plt.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=2,
        fancybox=True,
        shadow=True,
    )

    # Saving
    filename = f"{save_dir}/states/{csv_base}_state_{idx}.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

    ## Plot action
    plt.figure()
    plt.plot(actions, label=["Vd", "Vq"])
    plt.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=2,
        fancybox=True,
        shadow=True,
    )

    # Saving
    filename = f"{save_dir}/actions/{csv_base}_action_{idx}.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def plot_torque_tracking(idx, observations, actions, save_dir="plots", csv_base=""):
    PLT_STATE_DIR = os.path.join(save_dir, "torque_states")
    PLT_ACTION_DIR = os.path.join(save_dir, "torque_actions")

    os.makedirs(PLT_STATE_DIR, exist_ok=True)
    os.makedirs(PLT_ACTION_DIR, exist_ok=True)

    # Plot state
    plt.figure()
    plt.plot(observations, label=[r"$T_e$", r"$T_{e,ref}$", "Id", "Iq"])
    plt.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    filename = os.path.join(PLT_STATE_DIR, f"{csv_base}_torque_state_{idx}.png")
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

    # Plot action
    plt.figure()
    plt.plot(actions, label=["Vd", "Vq"])
    plt.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    filename = os.path.join(PLT_ACTION_DIR, f"{csv_base}_torque_action_{idx}.png")
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

def remove_outliers(errors):
    # Remove outliers from the error
    q1 = np.percentile(errors, 25, axis=0)
    q3 = np.percentile(errors, 75, axis=0)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    error_list = errors[np.all((errors >= lower_bound) & (errors <= upper_bound), axis=1)]
    return error_list


def plot_box(errors, save_dir="plots", csv_base=""):
    error_list = remove_outliers(errors)

    # Create a box plot of the error
    plt.figure(figsize=(8, 6))

    # Id error boxplot
    plt.boxplot(
        error_list,
        patch_artist=True,
        showmeans=True,
        showfliers=True,
        meanprops=dict(marker="D", markeredgecolor="black", markerfacecolor="white"),
        labels=["Id error", "Iq error"],
    )
    plt.ylabel("Absolute Error")
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    filename = f"{save_dir}/{csv_base}_boxplot.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved boxplot to {filename}")

def plot_boxes(error_dict, save_dir="plots"):
    colors = plt.cm.tab10.colors

    # --- Id error boxplot ---
    id_error_data = [remove_outliers(errors)[:, 0] for errors in error_dict.values()]
    fig_id, ax_id = plt.subplots(figsize=(7, 6))
    box1 = ax_id.boxplot(
        id_error_data,
        patch_artist=True,
        showmeans=True,
        showfliers=True,
        meanprops=dict(marker="D", markeredgecolor="black", markerfacecolor="white"),
        medianprops=dict(color="black"),
    )
    ax_id.set_ylabel("Id Error", fontsize=12)
    ax_id.set_xticks(range(1, len(error_dict) + 1))
    ax_id.set_xticklabels(error_dict.keys(), rotation=15, fontsize=10)
    ax_id.grid(True, linestyle="--", alpha=0.4)
    for patch, color in zip(box1["boxes"], colors):
        patch.set_facecolor(color)
    plt.tight_layout()
    filename_id = os.path.join(save_dir, "id_error_boxplot.png")
    fig_id.savefig(filename_id, dpi=300)
    plt.show()
    print(f"Saved Id error boxplot to {filename_id}")

    # --- Iq error boxplot ---
    iq_error_data = [remove_outliers(errors)[:, 1] for errors in error_dict.values()]
    fig_iq, ax_iq = plt.subplots(figsize=(7, 6))
    box2 = ax_iq.boxplot(
        iq_error_data,
        patch_artist=True,
        showmeans=True,
        showfliers=True,
        meanprops=dict(marker="D", markeredgecolor="black", markerfacecolor="white"),
        medianprops=dict(color="black"),
    )
    ax_iq.set_ylabel("Iq Error", fontsize=12)
    ax_iq.set_xticks(range(1, len(error_dict) + 1))
    ax_iq.set_xticklabels(error_dict.keys(), rotation=15, fontsize=10)
    ax_iq.grid(True, linestyle="--", alpha=0.4)
    for patch, color in zip(box2["boxes"], colors):
        patch.set_facecolor(color)
    plt.tight_layout()
    filename_iq = os.path.join(save_dir, "iq_error_boxplot.png")
    fig_iq.savefig(filename_iq, dpi=300)
    plt.show()
    print(f"Saved Iq error boxplot to {filename_iq}")

def plot_te_box(error_dict, save_dir="plots"):
    """
    Plots boxplot of torque errors (Te - Te_ref) for different models.
    """
    os.makedirs(save_dir, exist_ok=True)

    colors = plt.cm.tab10.colors

    # Remove outliers and flatten if needed
    data = [remove_outliers(errors).ravel() for errors in error_dict.values()]

    plt.figure(figsize=(8, 6))
    bp = plt.boxplot(
        data,
        patch_artist=True,
        showmeans=True,
        showfliers=True,
        meanprops=dict(marker="D", markeredgecolor="black", markerfacecolor="white"),
        medianprops=dict(color="black"),
    )

    # Apply colors
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    plt.ylabel("Torque Error (|Te - Te_ref|)", fontsize=12)
    plt.xticks(range(1, len(error_dict) + 1), error_dict.keys(), rotation=15, fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    filename = os.path.join(save_dir, "te_error_boxplot.png")
    plt.savefig(filename, dpi=300)
    plt.show()
    print(f"Saved torque boxplot to {filename}")