import os
import pandas as pd
import matplotlib.pyplot as plt

def compare_models_boxplot(csv_files, model_labels, save_dir="plots", last_n_steps=100):
    os.makedirs(save_dir, exist_ok=True)

    id_error_groups = []
    iq_error_groups = []

    for csv_file, label in zip(csv_files, model_labels):
        if not os.path.exists(csv_file):
            print(f"File not found: {csv_file}")
            continue

        df = pd.read_csv(csv_file)
        df['Id_error'] = abs(df['Id'] - df['Id_ref'])
        df['Iq_error'] = abs(df['Iq'] - df['Iq_ref'])

        # Collect last N steps from each episode
        last_id_errors = []
        last_iq_errors = []

        for ep, group in df.groupby('episode'):
            last_group = group.tail(last_n_steps)
            last_id_errors.extend(last_group['Id_error'].values)
            last_iq_errors.extend(last_group['Iq_error'].values)

        id_error_groups.append(last_id_errors)
        iq_error_groups.append(last_iq_errors)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Id error comparison
    axs[0].boxplot(id_error_groups, patch_artist=True, showmeans=True, showfliers=False,
                   meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='white'))
    axs[0].set_title(f'Absolute Id Error Comparison (Last {last_n_steps} Steps)')
    axs[0].set_ylabel('Id Error')
    axs[0].set_xticks(range(1, len(model_labels) + 1))
    axs[0].set_xticklabels(model_labels, rotation=15)
    axs[0].grid(True, linestyle='--', alpha=0.4)

    # Iq error comparison
    axs[1].boxplot(iq_error_groups, patch_artist=True, showmeans=True, showfliers=False,
                   meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='white'))
    axs[1].set_title(f'Absolute Iq Error Comparison (Last {last_n_steps} Steps)')
    axs[1].set_ylabel('Iq Error')
    axs[1].set_xticks(range(1, len(model_labels) + 1))
    axs[1].set_xticklabels(model_labels, rotation=15)
    axs[1].grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    filename = os.path.join(save_dir, "multi_model_error_comparison.png")
    plt.savefig(filename, dpi=300)
    plt.show()
    print(f"Saved comparison plot to {filename}")


if __name__ == "__main__":
    csv_files = [
        "test_results/MAC_1.csv",
        "test_results/SAC_1.csv"
    ]
    model_labels = [
        "MB-AC",
        "SAC"
    ]
    compare_models_boxplot(csv_files, model_labels, save_dir="plots", last_n_steps=100)
