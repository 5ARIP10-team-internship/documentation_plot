# load_model.py

import gymnasium as gym
import numpy as np
from environment import EnvPMSM
from mb2 import MBMFAlgorithm
from stable_baselines3 import SAC

def load_trained_model(model_type, reward_function="absolute", model_path=None):
    sys_params_dict = {
        "dt": 1 / 10e3,
        "r": 29.0808e-3,
        "ld": 0.91e-3,
        "lq": 1.17e-3,
        "lambda_PM": 0.172312604,
        "vdc": 1200,
        "we_nom": 200 * 2 * np.pi,
        "i_max": 200,
        "reward": reward_function,
    }

    env = EnvPMSM(sys_params=sys_params_dict)
    env = gym.wrappers.TimeLimit(env, 200)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    if model_type == "MBAC": # Model Based Actor-Critic
        model = MBMFAlgorithm(
            env,
            gamma=0.99,
            lr=1e-3,
            buffer_size=1000,
            batch_size=64,
            model_batch_size=64,
            model_planning_steps=2,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=500,
        )
        if model_path is None:
            model_path = f"models/EnvPMSM_{reward_function}_best.pth"
            print(f"No model_path provided, using default: {model_path}")
        else:
            print(f"Loading MBAC model from: {model_path}")
        model.load(model_path)
        use_stable_baselines = False

    elif model_type == "SAC":
        if model_path is None:
            raise ValueError("SAC requires a model_path (.zip) to load.")
        else:
            print(f"Loading SAC model from: {model_path}")
        model = SAC.load(model_path, env=env)
        use_stable_baselines = True

    #elif model_type == "TDMPC":

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model, env, sys_params_dict, use_stable_baselines



