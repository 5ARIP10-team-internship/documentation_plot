import argparse
import os

import numpy as np
import pandas as pd
from utils import plot_box, plot_three_phase

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")


def three_phase(csv_file, save_dir="plots", csv_base="", max_episodes=10):
    if not os.path.exists(csv_file):
        print(f" File not found: {csv_file}")
        return

    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(csv_file)

    episodes = df["episode"].unique()

    for i, ep in enumerate(episodes):
        if i >= max_episodes:
            break
        ep_data = df[df["episode"] == ep]
        states = ep_data[["Id", "Iq", "Id_ref", "Iq_ref"]].values
        actions = ep_data[["action_d", "action_q"]].values

        plot_three_phase(ep, states, actions, save_dir, csv_base)


def box(csv_file, save_dir="plots", csv_base="", last_n_steps=100):
    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        return

    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(csv_file)

    episodes = df["episode"].unique()
    errors = np.zeros((len(episodes), 2))
    for i, ep in enumerate(episodes):
        ep_data = df[df["episode"] == ep]
        Inorm_last = ep_data[["Id", "Iq"]].values[-last_n_steps:]
        Iref_last = ep_data[["Id_ref", "Iq_ref"]].values[-last_n_steps:]
        errors[i, :] = np.abs(np.mean(Inorm_last, axis=0) - Iref_last[-1, :])

    plot_box(errors, save_dir, csv_base)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to the test result CSV file")
    parser.add_argument(
        "--plot_type",
        type=str,
        choices=["error_rate", "box", "three_phase"],
        default="box",
        help="Type of plot: 'episode' for per-episode curves, 'box' for error distribution",
    )
    parser.add_argument("--max_episodes", type=int, default=10, help="Limit number of episodes to plot")
    parser.add_argument("--output_dir", type=str, default="plots", help="Directory to save plots")
    args = parser.parse_args()

    csv_file = os.path.join(DATA_DIR, args.csv)
    csv_base = os.path.splitext(os.path.basename(csv_file))[0]
    out_dir = os.path.join(ROOT_DIR, args.output_dir)

    if args.plot_type == "three_phase":
        three_phase(csv_file, out_dir, csv_base)
    elif args.plot_type == "box":
        box(csv_file, out_dir, csv_base)
