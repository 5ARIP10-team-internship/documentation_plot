import argparse
import os

import numpy as np
import pandas as pd
from utils import plot_boxes, plot_te_box

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")

def compare_models(csv_files, model_labels, save_dir="plots", last_n_steps=10, env_name="current"):
    os.makedirs(save_dir, exist_ok=True)

    error_dict = {}

    for csv_file, label in zip(csv_files, model_labels):
        if not os.path.exists(csv_file):
            print(f"File not found: {csv_file}")
            continue

        df = pd.read_csv(csv_file)
        episodes = df["episode"].unique()

        if env_name == "current":
            errors = np.zeros((len(episodes), 2))
            for i, ep in enumerate(episodes):
                ep_data = df[df["episode"] == ep]
                Inorm_last = ep_data[["Id", "Iq"]].values[-last_n_steps:]
                Iref_last = ep_data[["Id_ref", "Iq_ref"]].values[-last_n_steps:]
                errors[i, :] = np.abs(np.mean(Inorm_last, axis=0) - Iref_last[-1, :])

        elif env_name == "torque":
            errors = np.zeros((len(episodes), 1))
            for i, ep in enumerate(episodes):
                ep_data = df[df["episode"] == ep]
                Tnorm_last = ep_data[["Te"]].values[-last_n_steps:]
                Tref_last = ep_data[["Te_ref"]].values[-last_n_steps:]
                errors[i, :] = np.abs(np.mean(Tnorm_last, axis=0) - Tref_last[-1, :])
        else:
            raise ValueError(f"Unsupported env_name: {env_name}")

        error_dict[label] = errors

    if env_name == "current":
        plot_boxes(error_dict, save_dir)
    else:
        plot_te_box(error_dict, save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="plots", help="Directory to save plots")
    parser.add_argument("--env_name", type=str, choices=["current", "torque"], required=True, help="Environment type: current or torque")
    args = parser.parse_args()

    out_dir = os.path.join(ROOT_DIR, args.output_dir)

    csv_files = [
        # "pmsm_absolute.csv",
        # "pmsm_quadratic.csv",
        # "pmsm_final_square_root.csv",
        # "pmsm_quartic_root.csv",
        # "tcpmsm_absolute.csv",
        # "tcpmsm_quadratic.csv",
        # "tcpmsm_square_root.csv",
        "tcpmsm_final_square_root.csv",
        # "tcpmsm_quartic_root.csv",
        "SAC_torque_1000eps.csv",
        # "SAC_current_1000eps.csv",
    ]
    model_labels = [
                    # "Absolute",
                    # "Quadratic",
                    # "Square Root",
                    # "Quartic Root",
                    "TDMPC-square_root",
                    "SAC-absolute"
                    ]

    csv_files = [os.path.join(DATA_DIR, f) for f in csv_files]

    compare_models(csv_files, model_labels, save_dir=out_dir, env_name=args.env_name)
