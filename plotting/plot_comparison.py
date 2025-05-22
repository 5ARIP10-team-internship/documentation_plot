import argparse
import os

import numpy as np
import pandas as pd
from utils import plot_boxes

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")


def compare_models(csv_files, model_labels, save_dir="plots", last_n_steps=10):
    os.makedirs(save_dir, exist_ok=True)

    error_dict = {}

    for csv_file, label in zip(csv_files, model_labels):
        if not os.path.exists(csv_file):
            print(f"File not found: {csv_file}")
            continue

        df = pd.read_csv(csv_file)

        episodes = df["episode"].unique()
        errors = np.zeros((len(episodes), 2))
        for i, ep in enumerate(episodes):
            ep_data = df[df["episode"] == ep]
            Inorm_last = ep_data[["Id", "Iq"]].values[-last_n_steps:]
            Iref_last = ep_data[["Id_ref", "Iq_ref"]].values[-last_n_steps:]
            errors[i, :] = np.abs(np.mean(Inorm_last, axis=0) - Iref_last[-1, :])

        error_dict[label] = errors

    plot_boxes(error_dict, save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="plots", help="Directory to save plots")
    args = parser.parse_args()

    out_dir = os.path.join(ROOT_DIR, args.output_dir)

    csv_files = [
        "array_11872881_experiment_1.csv",
        "array_11872881_experiment_2.csv",
        "array_11872881_experiment_3.csv",
        "array_11872881_experiment_4.csv",
    ]
    model_labels = ["Absolute", "Quadratic", "Quadratic root", "Quartic root"]

    csv_files = [os.path.join(DATA_DIR, f) for f in csv_files]

    compare_models(csv_files, model_labels, save_dir=out_dir)
