import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os

import sys
import wandb
import argparse
from environment import *
from mb2 import MBMFAlgorithm
import pandas as pd



os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# CLI Input
parser = argparse.ArgumentParser()
parser.add_argument("--env_name", type=str, default="PMSM",
                    choices=['LoadRL', 'Load3RL', 'PMSM'], help='Environment name')
parser.add_argument("--reward_function", type=str, default="quadratic",
                    choices=['absolute', 'quadratic', 'quadratic_2', 'square_root', 'square_root_2',
                             'quartic_root', 'quartic_root_2'], help='Reward function type')
parser.add_argument("--job_id", type=str, default="")
parser.add_argument("--train", action="store_true", help="Enable training")
parser.add_argument("--test", action="store_true", help="Enable testing")
parser.add_argument("--epochs", type=int, default=2000, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for rewards")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for training")
parser.add_argument("--buffer_size", type=int, default=500000, help="Replay buffer size")
parser.add_argument("--model_batch_size", type=int, default=128, help="Batch size for model training")
parser.add_argument("--model_planning_steps", type=int, default=3, help="Number of model planning steps")
parser.add_argument("--epsilon_start", type=float, default=1.0, help="Starting epsilon value for epsilon-greedy")
parser.add_argument("--epsilon_end", type=float, default=0.05, help="Ending epsilon value for epsilon-greedy")
parser.add_argument("--epsilon_decay", type=int, default=500, help="Decay steps for epsilon value")

args = parser.parse_args()
env_name        = args.env_name
reward_function = args.reward_function
job_id          = args.job_id
train           = args.train
test            = args.test
epochs         = args.epochs
batch_size     = args.batch_size
gamma          = args.gamma
lr             = args.lr
buffer_size    = args.buffer_size
model_batch_size = args.model_batch_size
model_planning_steps = args.model_planning_steps
epsilon_start = args.epsilon_start
epsilon_end   = args.epsilon_end
epsilon_decay = args.epsilon_decay

#Set WandB project
wandb.init(
    project="PMSM-model-test",  # Project name in WandB
    name=f"Basic2.0_results_{epochs}",  # This run's name
    sync_tensorboard=False,  #Stop syncing tensorboard
    monitor_gym=True,  # Monitor Gym environment
    save_code=True,
)
log_dir = "wandb_logs/"
os.makedirs(log_dir, exist_ok=True)

if env_name == "LoadRL":
    if reward_function in ["quadratic_2", "square_root_2", "quartic_root_2"]:
        sys.exit("This reward function has not been implemented for this environment")
    sys_params_dict = {"dt": 1 / 10e3,  # Sampling time [s]
                       "r": 1,          # Resistance [Ohm]
                       "l": 1e-2,       # Inductance [H]
                       "vdc": 500,      # DC bus voltage [V]
                       "i_max": 100,    # Maximum current [A]
                       }
elif env_name == "Load3RL":
    sys_params_dict = {"dt": 1 / 10e3,      # Sampling time [s]
                       "r": 1,              # Resistance [Ohm]
                       "l": 1e-2,           # Inductance [H]
                       "vdc": 500,          # DC bus voltage [V]
                       "we_nom": 200*2*np.pi, # Nominal speed [rad/s]
                       }
    idq_max_norm = lambda vdq_max,we,r,l: vdq_max / np.sqrt(np.power(r, 2) + np.power(we * l, 2))
    # Maximum current [A]
    sys_params_dict["i_max"] = idq_max_norm(sys_params_dict["vdc"]/2, sys_params_dict["we_nom"],
                                            sys_params_dict["r"], sys_params_dict["l"])
elif env_name == "PMSM":
    sys_params_dict = {"dt": 1 / 10e3,      # Sampling time [s]
                       "r": 29.0808e-3,     # Resistance [Ohm]
                       "ld": 0.91e-3,       # Inductance d-frame [H]
                       "lq": 1.17e-3,       # Inductance q-frame [H]
                       "lambda_PM": 0.172312604, # Flux-linkage due to permanent magnets [Wb]
                       "vdc": 1200,             # DC bus voltage [V]
                       "we_nom": 200*2*np.pi,   # Nominal speed [rad/s]
                       "i_max": 200,            # Maximum current [A]
                       }
else:
    raise NotImplementedError
    # sys.exit("Environment name not existant")

environments = {"LoadRL": {"env": EnvLoadRL,
                    "name": f"Single Phase RL system / Delta Vdq penalty / Reward {reward_function}",
                    "max_episode_steps": 500,
                    "max_episodes": 200,
                    "reward": reward_function,
                    "model_name": f"EnvLoadRL_{reward_function}"
                    },
                "Load3RL": {"env": EnvLoad3RL,
                    "name": f"Three-phase RL system / Delta Vdq penalty / Reward {reward_function}",
                    "max_episode_steps": 500,
                    "max_episodes": 300,
                    "reward": reward_function,
                    "model_name": f"EnvLoad3RL_{reward_function}"
                    },
                "PMSM": {"env": EnvPMSM,
                    "name": f"PMSM / Delta Vdq penalty / Reward {reward_function}",
                    "max_episode_steps": 200,
                    "max_episodes": 300,
                    "reward": reward_function,
                    "model_name": f"EnvPMSM_{reward_function}"
                    },
                }

env_sel = environments[env_name]                # Choose Environment


sys_params_dict["reward"] = env_sel["reward"] 
env = env_sel["env"](sys_params=sys_params_dict)




# ----------------- 测试部分 -----------------
if test:
    # 强制测试使用 PMSM 环境
    env_name = "PMSM"
    # PMSM 的系统参数
    sys_params_dict = {
        "dt": 1 / 10e3,         # 采样时间 [s]
        "r": 29.0808e-3,        # 电阻 [Ohm]
        "ld": 0.91e-3,          # d 轴电感 [H]
        "lq": 1.17e-3,          # q 轴电感 [H]
        "lambda_PM": 0.172312604,  # 永磁体磁链 [Wb]
        "vdc": 1200,            # DC 总线电压 [V]
        "we_nom": 200 * 2 * np.pi,   # 标称速度 [rad/s]
        "i_max": 200,           # 最大电流 [A]
    }
    
    env_sel = environments[env_name]
    sys_params_dict["reward"] = env_sel["reward"]
    env = env_sel["env"](sys_params=sys_params_dict)
    env = gym.wrappers.TimeLimit(env, env_sel["max_episode_steps"])
    env = gym.wrappers.RecordEpisodeStatistics(env)

    # 与训练时相同的模型文件名
    best_model_path = "models/model.pth"

    # 加载模型
    model = MBMFAlgorithm(
        env, 
        gamma=gamma, 
        lr=lr, 
        buffer_size=buffer_size, 
        batch_size=batch_size,
        model_batch_size=model_batch_size, 
        model_planning_steps=model_planning_steps,
        epsilon_start=epsilon_start, 
        epsilon_end=epsilon_end, 
        epsilon_decay=epsilon_decay
    )
    model.load(best_model_path)

    plot = PlotTest()

    print(f"Testing: {env_sel['name']}")
    print(f"Model: {env_sel['model_name']} (loaded)")

    test_max_episodes = 1000
    id_errors_all = []
    iq_errors_all = []

    for episode in range(test_max_episodes):
        obs, info = env.reset(options={"Idref": 0, "Iqref": 100})
        # 将obs转为float32数组 (7维)
        full_state = np.array(obs, dtype=np.float32)
        # 只取前4个信号 (Id, Iq, Idref, Iqref) 用于绘图
        state = full_state[0:4]
        state_list = [state]
        action_list = []
        reward_list = []

        plt.figure(episode, figsize=(10, 6))
        done = False
        while not done:
            # 使用完整状态选取动作
            action = model.select_action(full_state, noise=False)
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, done, truncated, info = step_result
                done = done or truncated
            else:
                obs, reward, done, info = step_result
            full_state = np.array(obs, dtype=np.float32)
            # 取前4个信号 (Id, Iq, Idref, Iqref) 用于记录/绘图
            state = full_state[0:4]
            state_list.append(state)
            action_list.append(action)
            reward_list.append(reward)

            id_error = state[0] - state[2]
            iq_error = state[1] - state[3]
            id_errors_all.append(id_error)
            iq_errors_all.append(iq_error)

            wandb.log({
                f"epoch_{epochs}": episode+1,
                "id_error": id_error,
                "iq_error": iq_error,
            })

            if done:
                break

        # 如果有 PlotTest, 可以调用:
        name = f"MB2.0_{epochs}"
        plot.plot_three_phase(
            episode, state_list, action_list, reward_list,
            env_sel['model_name'], env_sel['reward'], name, sys_params_dict['we_nom'] * full_state[4]
        )

    # 在所有episode之后计算平均误差并打印
    mean_id_error = np.mean(np.abs(id_errors_all))
    mean_iq_error = np.mean(np.abs(iq_errors_all))
    print(f"Average Id error: {mean_id_error:.4f}")
    print(f"Average Iq error: {mean_iq_error:.4f}")
    df = pd.DataFrame({'abs_id_error': np.abs(id_errors_all)})
    df.to_csv(f"MBAC2n_{env_sel['reward']}_{epochs}_id_errors.csv", index=False)
    df = pd.DataFrame({"abs_iq_error": np.abs(iq_errors_all)})
    df.to_csv(f"MBAC2n_{env_sel['reward']}_{epochs}_iq_errors.csv", index=False)

    env.close()
    sys.exit()
