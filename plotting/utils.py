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

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Id error boxplot
    id_error_data = [remove_outliers(errors)[:, 0] for errors in error_dict.values()]
    box1 = axs[0].boxplot(
        id_error_data,
        patch_artist=True,
        showmeans=True,
        showfliers=True,
        meanprops=dict(marker="D", markeredgecolor="black", markerfacecolor="white"),
        medianprops=dict(color="black"),
    )
    axs[0].set_ylabel("Id Error", fontsize=12)
    axs[0].set_xticks(range(1, len(error_dict) + 1))
    axs[0].set_xticklabels(error_dict.keys(), rotation=15, fontsize=10)
    axs[0].grid(True, linestyle="--", alpha=0.4)

    # Iq error boxplot
    iq_error_data = [remove_outliers(errors)[:, 1] for errors in error_dict.values()]
    box2 = axs[1].boxplot(
        iq_error_data,
        patch_artist=True,
        showmeans=True,
        showfliers=True,
        meanprops=dict(marker="D", markeredgecolor="black", markerfacecolor="white"),
        medianprops=dict(color="black"),
    )
    axs[1].set_ylabel("Iq Error", fontsize=12)
    axs[1].set_xticks(range(1, len(error_dict) + 1))
    axs[1].set_xticklabels(error_dict.keys(), rotation=15, fontsize=10)
    axs[1].grid(True, linestyle="--", alpha=0.4)

    # Set colors for boxplots
    for patch, color in zip(box1["boxes"], colors):
        patch.set_facecolor(color)
    for patch, color in zip(box2["boxes"], colors):
        patch.set_facecolor(color)

    plt.tight_layout()
    filename = os.path.join(save_dir, "comparison_boxplot.png")
    plt.savefig(filename, dpi=300)
    plt.show()
    print(f"Saved boxplot to {filename}")
