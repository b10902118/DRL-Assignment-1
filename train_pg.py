import importlib.util
from env import DynamicTaxiEnv
import student_agent
from student_agent import (
    get_action,
    get_state,
    post_update_internal_state,
    manhattan_distance,
    SIGHT,
)
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
from itertools import product

"""
state:
    has_passenger
    closest_possible_station_relative_position
    obstacles
    near_passenger
    near_destionation

training:
    reward shaping: panalize not moving closer

TODO: try policy gradient
TODO: draw reward curve
Not use clipping for correct bootstrapping?
"""

N_ACTIONS = 6

LOG_EPISODES = 1000


def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Numeric stability
    return exp_x / exp_x.sum()


def get_init_tables(sight=SIGHT):
    policy_table = {"Done": np.zeros(N_ACTIONS)}
    update_count = {"Done": np.zeros(N_ACTIONS)}
    for has_passenger in [True, False]:
        for rel_pos in product(range(-sight, sight + 1), repeat=2):
            for obstacles in product([0, 1], repeat=4):
                for near in [True, False]:
                    md = manhattan_distance(rel_pos, (0, 0))
                    if (md <= 1 and not near) or (md > 1 and near):
                        continue
                    state = (has_passenger, rel_pos, obstacles, near)

                    # expected_reward = 50 * 0.99**md - 0.1 * md
                    entry = np.zeros(N_ACTIONS)

                    # encourage moving closer
                    rel_pos_sign = np.sign(rel_pos)
                    entry[student_agent.MOVENORTH] -= 0.1 * rel_pos_sign[0]
                    entry[student_agent.MOVESOUTH] += 0.1 * rel_pos_sign[0]
                    entry[student_agent.MOVEEAST] += 0.1 * rel_pos_sign[1]
                    entry[student_agent.MOVEWEST] -= 0.1 * rel_pos_sign[1]

                    for i, ob in enumerate(obstacles):
                        if ob:
                            entry[i] = -5

                    if md != 0:
                        entry[student_agent.PICKUP] -= 10
                        entry[student_agent.DROPOFF] -= 10
                    else:
                        if has_passenger:
                            entry[student_agent.DROPOFF] += 50
                        else:
                            entry[student_agent.PICKUP] += 50

                    policy_table[state] = entry
                    update_count[state] = np.zeros(N_ACTIONS)
    # pickle.dump(policy_table, open("q_table.pkl", "wb"))
    # pickle.dump(update_count, open("update_count.pkl", "wb"))
    return policy_table, update_count


def train(
    grid_sizes,
    alpha=0.0001,
    gamma=0.99,
):
    episodes = len(grid_sizes)
    policy_table, update_count = get_init_tables()

    alpha_start = alpha
    rewards_per_episode = []
    for episode, grid_size in enumerate(grid_sizes):
        env = DynamicTaxiEnv(
            **{
                "grid_size": grid_size,
                "fuel_limit": 5000,
                "randomize_passenger": True,
                "randomize_destination": True,
            }
        )
        student_agent.initialized = False
        obs, _ = env.reset()
        total_reward = 0
        done = False
        step_count = 0

        state = get_state(obs)

        shaped_reward = 0
        trajectory = []
        while not done:
            prev_picked_up = env.passenger_picked_up
            action = np.random.choice(N_ACTIONS, p=softmax(policy_table[state]))
            student_agent.post_update_internal_state(state, action)
            obs, reward, done, _ = env.step(action)

            next_state = get_state(obs)

            # reward shaping
            if next_state != "Done":
                # reward shaping for nearest station
                closest_relative_pos = state[1]
                next_closest_relative_pos = next_state[1]
                if manhattan_distance(
                    next_closest_relative_pos, (0, 0)
                ) >= manhattan_distance(closest_relative_pos, (0, 0)):
                    reward -= 1
                    shaped_reward -= 1

                # reward shaping for picking up passenger
                if (
                    action == student_agent.PICKUP
                    and not prev_picked_up
                    and env.passenger_picked_up
                ):
                    reward += 10
                    shaped_reward += 10

                # reward shaping for dropping off passenger (not at destination)
                # heavier so won't drop and pick
                if action == student_agent.DROPOFF:
                    reward -= 20
                    shaped_reward -= 20

            trajectory.append((state, action, reward))

            state = next_state
            total_reward += reward
            step_count += 1

        # ✅ **Policy Update (REINFORCE-like)**
        G = 0  # Return (discounted sum of rewards)
        for t in reversed(range(len(trajectory))):
            state, action, reward = trajectory[t]
            G = reward + gamma * G  # Discounted reward

            # TODO: Update policy table using policy gradient
            logit = softmax(policy_table[state])
            g = (
                1 - logit[action]
            )  # np.array([-p if i != action else 1 - p for i, p in enumerate(logit) ])
            policy_table[state][action] += alpha * gamma**t * G * g
            update_count[state][action] += 1

        rewards_per_episode.append(total_reward - shaped_reward)
        alpha = alpha_start * (1 - episode / episodes)

        if (episode + 1) % LOG_EPISODES == 0:
            avg_reward = np.mean(rewards_per_episode[-LOG_EPISODES:])
            # max_reward = np.max(rewards_per_episode[-LOG_EPISODES:])
            # max_q_value = np.max([np.max(values) for values in q_table.values()])
            print(
                f"🚀 Episode {episode + 1}/{episodes}, Average Reward: {avg_reward:.2f}"
            )
        # print(f"{i}: Agent Finished in {step_count} steps, Score: {total_reward}")

    # Plot rewards per episode
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_per_episode[-100000:], label="Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Rewards per Episode")
    plt.legend()
    plt.grid()
    plt.savefig(f"{filename}.png")
    return policy_table, update_count


if len(sys.argv) > 1:
    filename = sys.argv[1]
    if not filename.endswith(".pkl"):
        filename += ".pkl"
else:
    filename = "q_table_pg.pkl"

print(f"Training agent and saving Q-table to {filename}")

grid_sizes = [4] * 100000 + [6] * 100000 + [8] * 100000 + [9] * 100000 + [10] * 100000
# + ([5] * 50000 + [6] * 50000) * 6  # + ([8] * 50000 + [10] * 50000) * 6
# grid_sizes = (
#    [4] * 15000 + [5] * 15000 + [6] * 10000 + [8] * 10000 + [10] * 10000 + [15] * 10000
# )
qtable, update_count = train(grid_sizes)
with open(filename, "wb") as f:
    pickle.dump(qtable, f)
with open(f"{filename}_count.pkl", "wb") as f:
    pickle.dump(update_count, f)
