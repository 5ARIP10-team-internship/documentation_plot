# test_model.py
import pandas as pd
import numpy as np
import argparse
import os
import wandb
from load_model import load_trained_model
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


random.seed(42)
def test_and_save_csv(model_type, model_path, reward_function, num_episodes, test_id):
    model, env, sys_params_dict, use_stable_baselines = load_trained_model(
        model_type, reward_function, model_path
    )

    data_records = []

    for episode in range(num_episodes):
        obs, _ = env.reset(options={"Idref": 0, "Iqref": 100})
        step = 0
        done = False
        while not done:
            if use_stable_baselines:
                action, _ = model.predict(obs, deterministic=True) # deterministic=True for SAC
            else:
                full_state = np.array(obs, dtype=np.float32) # Convert to float32
                action = model.select_action(full_state, noise=False)

            obs, reward, done, truncated, info = env.step(action)
            done = done or truncated

            state = obs[0:4]
            if len(obs) > 4:
                speed = sys_params_dict['we_nom'] * obs[4] # Speed data for plotTest
            else:
                speed = None

            record = {
                "episode": episode,
                "step": step,
                "Id": state[0],
                "Iq": state[1],
                "Id_ref": state[2],
                "Iq_ref": state[3],
                "action_d": action[0],
                "action_q": action[1],
                "reward": reward,
                 "speed": speed
            }
            data_records.append(record)
            step += 1

    df = pd.DataFrame(data_records)

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    output_dir = "test_results"
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, f"{model_name}_{test_id}.csv")

    df.to_csv(output_csv, index=False)
    print(f"Saved test data to {output_csv}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, choices=["MBMF", "SAC"], help="Type of model to test")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file")
    parser.add_argument("--reward_function", type=str, default="absolute", help="Reward function used")
    parser.add_argument("--episodes", type=int, default=100, help="Number of test episodes to run")
    parser.add_argument("--ID", type=str, required=True, help="Test identifier to include in output filename")
    args = parser.parse_args()

    test_and_save_csv(
        model_type=args.model_type,
        model_path=args.model_path,
        reward_function=args.reward_function,
        num_episodes=args.episodes,
        test_id=args.ID
    )
