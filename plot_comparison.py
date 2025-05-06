import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def remove_outliers(data):
    data = np.array(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered = data[(data >= lower_bound) & (data <= upper_bound)]
    return filtered

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

        last_id_errors = []
        last_iq_errors = []

        for ep, group in df.groupby('episode'):
            last_group = group.tail(last_n_steps)
            last_id_errors.extend(last_group['Id_error'].values)
            last_iq_errors.extend(last_group['Iq_error'].values)

        # Apply outlier removal
        filtered_id_errors = remove_outliers(last_id_errors)
        filtered_iq_errors = remove_outliers(last_iq_errors)

        id_error_groups.append(filtered_id_errors)
        iq_error_groups.append(filtered_iq_errors)

    colors = plt.cm.Set3.colors

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Id error comparison
    box1 = axs[0].boxplot(id_error_groups, patch_artist=True, showmeans=True, showfliers=True,
                          meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='white'))
    axs[0].set_title(f'Absolute Id Error Comparison (Last {last_n_steps} Steps, Filtered)', fontsize=14)
    axs[0].set_ylabel('Id Error', fontsize=12)
    axs[0].set_xticks(range(1, len(model_labels) + 1))
    axs[0].set_xticklabels(model_labels, rotation=15, fontsize=10)
    axs[0].grid(True, linestyle='--', alpha=0.4)

    for patch, color in zip(box1['boxes'], colors):
        patch.set_facecolor(color)

    # Iq error comparison
    box2 = axs[1].boxplot(iq_error_groups, patch_artist=True, showmeans=True, showfliers=True,
                          meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='white'))
    axs[1].set_title(f'Absolute Iq Error Comparison (Last {last_n_steps} Steps, Filtered)', fontsize=14)
    axs[1].set_ylabel('Iq Error', fontsize=12)
    axs[1].set_xticks(range(1, len(model_labels) + 1))
    axs[1].set_xticklabels(model_labels, rotation=15, fontsize=10)
    axs[1].grid(True, linestyle='--', alpha=0.4)

    for patch, color in zip(box2['boxes'], colors):
        patch.set_facecolor(color)

    plt.tight_layout()
    filename = os.path.join(save_dir, "multi_model_abs_error_comparison_filtered.png")
    plt.savefig(filename, dpi=300)
    plt.show()
    print(f"Saved comparison plot to {filename}")

def compare_models_error_rate_boxplot(csv_files, model_labels, save_dir="plots"): # Only calculate the last step error rate
    os.makedirs(save_dir, exist_ok=True)

    id_error_rate_groups = []
    iq_error_rate_groups = []

    epsilon = 1e-6

    for csv_file, label in zip(csv_files, model_labels):
        if not os.path.exists(csv_file):
            print(f"File not found: {csv_file}")
            continue

        df = pd.read_csv(csv_file)
        df['Id_error_rate'] = abs(df['Id'] - df['Id_ref']) / (abs(df['Id_ref']) + epsilon) * 100
        df['Iq_error_rate'] = abs(df['Iq'] - df['Iq_ref']) / (abs(df['Iq_ref']) + epsilon) * 100

        last_id_error_rates = []
        last_iq_error_rates = []

        for ep, group in df.groupby('episode'):
            last_row = group.tail(1)
            last_id_error_rates.append(last_row['Id_error_rate'].values[0])
            last_iq_error_rates.append(last_row['Iq_error_rate'].values[0])

        # Apply outlier removal
        filtered_id_error_rates = remove_outliers(last_id_error_rates)
        filtered_iq_error_rates = remove_outliers(last_iq_error_rates)

        id_error_rate_groups.append(filtered_id_error_rates)
        iq_error_rate_groups.append(filtered_iq_error_rates)

    colors = plt.cm.Set3.colors

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Id error rate boxplot
    box1 = axs[0].boxplot(id_error_rate_groups, patch_artist=True, showmeans=True, showfliers=True,
                          meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='white'))
    axs[0].set_title('Last Step Id Error Rate Comparison (Filtered, per Episode)', fontsize=14)
    axs[0].set_ylabel('Id Error Rate (%)', fontsize=12)
    axs[0].set_xticks(range(1, len(model_labels) + 1))
    axs[0].set_xticklabels(model_labels, rotation=15, fontsize=10)
    axs[0].grid(True, linestyle='--', alpha=0.4)

    for patch, color in zip(box1['boxes'], colors):
        patch.set_facecolor(color)

    # Iq error rate boxplot
    box2 = axs[1].boxplot(iq_error_rate_groups, patch_artist=True, showmeans=True, showfliers=True,
                          meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='white'))
    axs[1].set_title('Last Step Iq Error Rate Comparison (Filtered, per Episode)', fontsize=14)
    axs[1].set_ylabel('Iq Error Rate (%)', fontsize=12)
    axs[1].set_xticks(range(1, len(model_labels) + 1))
    axs[1].set_xticklabels(model_labels, rotation=15, fontsize=10)
    axs[1].grid(True, linestyle='--', alpha=0.4)

    for patch, color in zip(box2['boxes'], colors):
        patch.set_facecolor(color)

    plt.tight_layout()
    filename = os.path.join(save_dir, "multi_model_error_rate_comparison_filtered.png")
    plt.savefig(filename, dpi=300)
    plt.show()
    print(f"Saved last-step error rate comparison plot to {filename}")

if __name__ == "__main__":
    csv_files = [
        "test_results/MAC_20eps.csv",
        "test_results/SAC_20eps.csv"
    ]
    model_labels = [
        "MB-AC_100eps",
        "SAC_100eps"
    ]
    compare_models_boxplot(csv_files, model_labels, save_dir="plots", last_n_steps=20)
    #compare_models_error_rate_boxplot(csv_files, model_labels, save_dir="plots")
