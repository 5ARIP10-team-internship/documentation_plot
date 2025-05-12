import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from utils import PlotTest

# def plot_episodes(csv_file, save_dir="plots", max_episodes=None, csv_base=""):
#     if not os.path.exists(csv_file):
#         print(f" File not found: {csv_file}")
#         return
#
#     os.makedirs(save_dir, exist_ok=True)
#     df = pd.read_csv(csv_file)
#     episodes = df['episode'].unique()
#     if max_episodes:
#         episodes = episodes[:max_episodes]
#
#     for ep in episodes:
#         ep_data = df[df['episode'] == ep]
#
#         fig, axs = plt.subplots(1, 3, figsize=(15, 4))
#         fig.suptitle(f"Episode {ep}")
#
#         # Plot currents
#         axs[0].plot(ep_data['step'], ep_data['Id'], label='Id')
#         axs[0].plot(ep_data['step'], ep_data['Iq'], label='Iq')
#         axs[0].plot(ep_data['step'], ep_data['Id_ref'], '--', label='Id_ref')
#         axs[0].plot(ep_data['step'], ep_data['Iq_ref'], '--', label='Iq_ref')
#         axs[0].set_title('Currents')
#         axs[0].legend()
#
#         # Plot voltages
#         axs[1].plot(ep_data['step'], ep_data['action_d'], label='Vd')
#         axs[1].plot(ep_data['step'], ep_data['action_q'], label='Vq')
#         axs[1].set_title('Voltages')
#         axs[1].legend()
#
#         # Plot reward
#         axs[2].plot(ep_data['step'], ep_data['reward'])
#         axs[2].set_title('Reward')
#
#         plt.tight_layout()
#         filename = f"{save_dir}/{csv_base}_episode_{ep}.png"
#         plt.savefig(filename)
#         plt.close(fig)
#
#     print(f" Saved {len(episodes)} episode plots to {save_dir}/")

# def plot_box(csv_file, save_dir="plots", max_episodes=None, csv_base=""):
#     if not os.path.exists(csv_file):
#         print(f" File not found: {csv_file}")
#         return
#
#     os.makedirs(save_dir, exist_ok=True)
#     df = pd.read_csv(csv_file)
#
#     # Compute absolute error
#     df['Id_error'] = abs(df['Id'] - df['Id_ref'])
#     df['Iq_error'] = abs(df['Iq'] - df['Iq_ref'])
#
#     episodes = df['episode'].unique()
#     if max_episodes is not None:
#         try:
#             max_episodes = int(max_episodes)
#             episodes = episodes[:max_episodes]
#         except ValueError:
#             print(f" Invalid max_episodes value: {max_episodes}, ignored.")
#
#     for ep in episodes:
#         ep_data = df[df['episode'] == ep]
#
#         fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#         axs[0].boxplot(ep_data['Id_error'],patch_artist=True, showmeans=True, showfliers=False,
#                    meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='white'))
#         axs[0].set_title(f'Episode {ep} - Id Absolute Error')
#
#         axs[1].boxplot(ep_data['Iq_error'],patch_artist=True, showmeans=True, showfliers=False,
#                    meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='white'))
#         axs[1].set_title(f'Episode {ep} - Iq Absolute Error')
#
#         plt.tight_layout()
#         filename = f"{save_dir}/{csv_base}_episode_{ep}_boxplot.png"
#         plt.savefig(filename)
#         plt.close(fig)
#
#     print(f" Saved {len(episodes)} per-episode box plots to {save_dir}/")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def remove_outliers(data):
    data = np.array(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered = data[(data >= lower_bound) & (data <= upper_bound)]
    return filtered

def plot_box_single(csv_file, save_dir="plots", csv_base="", last_n_steps=100):
    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        return

    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(csv_file)

    # Compute absolute error
    df['Id_error'] = abs(df['Id'] - df['Id_ref'])
    df['Iq_error'] = abs(df['Iq'] - df['Iq_ref'])

    # Collect last N steps from each episode
    last_id_errors = []
    last_iq_errors = []

    for ep, group in df.groupby('episode'):
        last_group = group.tail(last_n_steps)
        last_id_errors.extend(last_group['Id_error'].values)
        last_iq_errors.extend(last_group['Iq_error'].values)

    # Remove outliers
    filtered_id_errors = remove_outliers(last_id_errors)
    filtered_iq_errors = remove_outliers(last_iq_errors)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Id error boxplot
    axs[0].boxplot(filtered_id_errors, patch_artist=True, showmeans=True, showfliers=True,
                   meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='white'))
    axs[0].set_title(f'Last {last_n_steps} Steps (Filtered): Id Absolute Error')
    axs[0].set_ylabel('Absolute Error')
    axs[0].set_xticks([1])
    axs[0].set_xticklabels(['Id Error'])
    axs[0].grid(True, linestyle='--', alpha=0.4)

    # Iq error boxplot
    axs[1].boxplot(filtered_iq_errors, patch_artist=True, showmeans=True, showfliers=True,
                   meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='white'))
    axs[1].set_title(f'Last {last_n_steps} Steps (Filtered): Iq Absolute Error')
    axs[1].set_xticks([1])
    axs[1].set_xticklabels(['Iq Error'])
    axs[1].grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    filename = f"{save_dir}/{csv_base}_last{last_n_steps}_error_boxplot_filtered.png"
    plt.savefig(filename, dpi=300)
    plt.show()
    print(f"Saved last {last_n_steps} steps filtered boxplot to {filename}")


def three_phase(csv_file, save_dir="plots", csv_base=""):
    if not os.path.exists(csv_file):
        print(f" File not found: {csv_file}")
        return

    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(csv_file)
    episodes = df['episode'].unique()

    plotter = PlotTest()
    env_name = "PMSM"
    reward_function = "quadratic"
    model_name = csv_base

    # for ep in episodes:
    ep = 16
    ep_data = df[df['episode'] == ep]
    states = ep_data[['Id', 'Iq', 'Id_ref', 'Iq_ref']].values
    actions = ep_data[['action_d', 'action_q']].values
    rewards = ep_data['reward'].values
    speed = ep_data['speed'].values[0] if 'speed' in ep_data.columns else None

    plotter.plot_three_phase(ep, states, actions, rewards, env_name, reward_function, model_name,speed)
    print(f" Saved three-phase plot for episode {ep}")







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to the test result CSV file")
    parser.add_argument("--plot_type", type=str, choices=["error_rate", "box", "three_phase"], default="box",
                        help="Type of plot: 'episode' for per-episode curves, 'box' for error distribution")
    parser.add_argument("--max_episodes", type=int, default=None, help="Limit number of episodes to plot")
    parser.add_argument("--output_dir", type=str, default="plots", help="Directory to save plots")
    args = parser.parse_args()

    # Extract base name without extension
    csv_base = os.path.splitext(os.path.basename(args.csv))[0]

    if args.plot_type == "three_phase":
        three_phase(args.csv, args.output_dir, csv_base)
    elif args.plot_type == "box":
        #plot_box(args.csv, args.output_dir, args.max_episodes, csv_base)
        plot_box_single(args.csv, args.output_dir, csv_base)
