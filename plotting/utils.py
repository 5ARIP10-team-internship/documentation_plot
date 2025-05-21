from matplotlib import pyplot as plt


class PlotTest:
    def __init__(self):
        return

    def plot_single_phase(
        self, idx, observations, actions, reward, env_name, reward_type
    ):
        plt.clf()
        plt.suptitle(f"Reward: {reward_type}\n")
        # Plot State
        ax = plt.subplot(131)
        ax.set_title("State vs step")
        ax.plot(observations, label=["I", "Iref"])
        box = ax.get_position()
        ax.set_position(
            [box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9]
        )
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.25),
            ncol=2,
            fancybox=True,
            shadow=True,
        )
        # Plot action
        ax = plt.subplot(132)
        ax.set_title("Action vs step")
        ax.plot(actions, label=["V"])
        box = ax.get_position()
        ax.set_position(
            [box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9]
        )
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.25),
            ncol=2,
            fancybox=True,
            shadow=True,
        )
        # Plot reward
        ax = plt.subplot(133)
        ax.set_title("Reward vs step")
        ax.plot(reward)
        box = ax.get_position()
        ax.set_position(
            [box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9]
        )

        plt.savefig(f"plots/{env_name}_{idx}.pdf", bbox_inches="tight")
        plt.pause(0.001)  # pause a bit so that plots are updated

    def plot_three_phase(
        self,
        idx,
        observations,
        actions,
        reward,
        env_name,
        reward_type,
        model_name,
        speed=None,
    ):
        fig = plt.figure(figsize=(12, 4))  # 明确使用一个 figure 对象

        if speed is not None:
            fig.suptitle(f"Reward: {reward_type}\nSpeed = {speed} [rad/s]")

        # Plot State
        ax1 = fig.add_subplot(131)
        ax1.set_title("State vs step")
        ax1.plot(observations, label=["Id", "Iq", "Idref", "Iqref"])
        ax1.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.3),
            ncol=2,
            fancybox=True,
            shadow=True,
        )

        # Plot Action
        ax2 = fig.add_subplot(132)
        ax2.set_title("Action vs step")
        ax2.plot(actions, label=["Vd", "Vq"])
        ax2.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.3),
            ncol=2,
            fancybox=True,
            shadow=True,
        )

        # Plot Reward
        ax3 = fig.add_subplot(133)
        ax3.set_title("Reward vs step")
        ax3.plot(reward)

        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        plt.savefig(f"plots/{env_name}_{model_name}_{idx}.pdf", bbox_inches="tight")
        plt.close(fig)
