import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output


def plot_trajectory(obs, label, color, reward):
    '''Plots the trajectory as a function of X and Y positions with additional details'''
    obs = np.array(obs)  # Converts the list into an array numpy
    x_pos = obs[:, 0]  # X coordinate
    y_pos = obs[:, 1]  # Y coordinate

    plt.plot(x_pos, y_pos, label=label, color=color, marker="o", linestyle="-")
    plt.scatter(x_pos[0], y_pos[0], color="yellow", edgecolors="k", s=120, marker="*", label=f"{label} Start")  # Début
    plt.scatter(x_pos[-1], y_pos[-1], color=color, edgecolors="k", s=100, label=f"{label} End")  # Fin


def get_human_choice(obs_1, obs_2, reward_1, reward_2, step, total_steps):
    """
    Displays two trajectories and asks the user to choose the best one.
    
    Parameters :
    - obs_1, obs_2: List of observations for each path.
    - reward_1, reward_2: Rewards associated with each path.
    - step, total_steps: Process progress indicators.
    """
    clear_output(wait=True)

    print(f"Trajectory Comparison {step}/{total_steps}")
    print(f"Rewards: 1st trajectory = {reward_1:.2f}, 2nd trajectory = {reward_2:.2f}")
    print("[0] Equally good")
    print("[1] The 1st trajectory is better")
    print("[2] The 2nd trajectory is better")
    print("[3] None are good")

    # Creation of trajectory graphs
    plt.figure(figsize=(7, 7))
    plot_trajectory(obs_1, "Trajectory 1", "blue", reward_1)
    plot_trajectory(obs_2, "Trajectory 2", "red", reward_2)

    plt.axhline(0, color="black", linestyle="--", label="Ground")  # Ground line
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("LunarLander Trajectories")
    plt.legend()
    plt.grid()
    plt.show()
    plt.close()

    # Checking user input
    while True:
        try:
            choice = int(input("Press 0, 1, 2, or 3: ").strip())
            if choice in {0, 1, 2, 3}:
                break
        except ValueError:
            pass
        print("Invalid input. Please enter 0, 1, 2, or 3.")

    # Matching choice with human feedback
    choice_map = {
        0: [0.5, 0.5],
        1: [1, 0],
        2: [0, 1],
        3: [0, 0]
    }

    return choice_map[choice]


class Space:
    def __init__(self, env, agent, real_human_check=False):
        self.real_human_check = real_human_check
        print("Initialize Space: Start")

        self.space = []

        while len(self.space) < 10:
            obs, _ = env.reset()
            time_step = 0
            obs_1_list = []
            obs_2_list = []
            real_reward_1 = 0
            real_reward_2 = 0

            obs = np.array(obs, dtype=np.float32)

            while len(self.space) < 10:
                time_step += 1

                action, _, _ = agent.act(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)

                next_obs = np.array(next_obs, dtype=np.float32)
                obs = next_obs

                if time_step <= 25:
                    obs_1_list.append(obs)  # Stores the state vector
                    real_reward_1 += reward
                else:
                    obs_2_list.append(obs)
                    real_reward_2 += reward

                if time_step >= 50:
                    if real_human_check:
                        human_choice = get_human_choice(
                            obs_1_list, obs_2_list, real_reward_1, real_reward_2, len(self.space), 10
                        )
                    else:
                        # Simulates the behaviour of a human assessor
                        human_choice = [1, 0] if real_reward_1 > real_reward_2 else [0, 1]

                    if human_choice != [0, 0]:
                        self.space.append(
                            np.array([obs_1_list, obs_2_list, human_choice], dtype=object)
                        )

                    obs_1_list = []
                    obs_2_list = []
                    time_step = 0

                if terminated or truncated:
                    break

        print("Initialize Space: End")

    def feeding_space(self, obs1, obs2, real_reward_1=[], real_reward_2=[], total_steps=20):
        if self.real_human_check:
            human_choice = get_human_choice(
                obs1, obs2, np.sum(real_reward_1), np.sum(real_reward_2), len(self.space), total_steps
            )
        else:
            human_choice = [1, 0] if np.sum(real_reward_1) > np.sum(real_reward_2) else [0, 1]

        if human_choice != [0, 0]:
            self.space.append(np.array([obs1, obs2, human_choice], dtype=object))

        # Keeps only the last 20 pairs of observations
        if len(self.space) > 20:
            self.space.pop(0)

        return self.space
