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


def plot_box(errors, save_dir="plots", csv_base=""):
    # Remove outliers from the error list
    q1 = np.percentile(errors, 25, axis=0)
    q3 = np.percentile(errors, 75, axis=0)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    error_list = errors[np.all((errors >= lower_bound) & (errors <= upper_bound), axis=1)]

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
