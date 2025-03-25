import importlib.util
from env import DynamicTaxiEnv

import time
import random
import os


def get_action_name(action):
    """Returns a human-readable action name."""
    actions = [
        "Move South",
        "Move North",
        "Move East",
        "Move West",
        "Pick Up",
        "Drop Off",
    ]
    return actions[action] if action is not None else "None"


def clear_page() -> None:
    """
    Used to clear the previously printed breakpoint information before
    printing the next information.
    """
    num_lines = os.get_terminal_size().lines
    for _ in range(num_lines):
        print()
    print("\033[0;0H")  # Ansi escape code: Set cursor to 0,0 position
    print("\033[J")  # Ansi escape code: Clear contents from cursor to end of screen


def render_env(self, action=None):
    clear_page()

    grid = [["• "] * self.grid_size for _ in range(self.grid_size)]

    """
    # Place passenger
    py, px = passenger_pos
    if 0 <= px < self.grid_size and 0 <= py < self.grid_size:
        grid[py][px] = 'P'
    """

    for k, v in self.station_map.items():
        grid[v[0]][v[1]] = k + " "

    grid[self.destination[0]][self.destination[1]] = "🚩"
    grid[self.passenger_loc[0]][self.passenger_loc[1]] = "👨"
    for obstacle in self.obstacles:
        grid[obstacle[0]][obstacle[1]] = "██"
    """
    # Place destination
    dy, dx = destination_pos
    if 0 <= dx < self.grid_size and 0 <= dy < self.grid_size:
        grid[dy][dx] = 'D'
    """
    # Place taxi
    ty, tx = self.taxi_pos
    if 0 <= tx < self.grid_size and 0 <= ty < self.grid_size:
        grid[ty][tx] = "🚖"

    # Print step info
    print(f"Taxi Position: ({tx}, {ty})")
    # print(f"Passenger Position: ({px}, {py}) {'(In Taxi)' if (px, py) == (tx, ty) else ''}")
    # print(f"Destination: ({dx}, {dy})")
    print("passenger_picked_up: ", self.passenger_picked_up)
    print("fuel: ", self.current_fuel)
    print(f"Last Action: {get_action_name(action)}\n")

    # Print grid
    for row in grid:
        print(" ".join(row))
    print("\n")


DynamicTaxiEnv.render = render_env


def run_agent(agent_file, grid_size, render=False):
    spec = importlib.util.spec_from_file_location("student_agent", agent_file)
    student_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_agent)

    env = DynamicTaxiEnv(
        **{
            "grid_size": grid_size,
            "fuel_limit": 150,
            "randomize_passenger": True,
            "randomize_destination": True,
            #   "render_modes": "ansi",
        }
    )
    print(f"Grid size: {grid_size}")
    obs, _ = env.reset()
    total_reward = 0
    done = False
    step_count = 0

    # print(env.__dict__)
    # exit()
    while not done:

        action = student_agent.get_action(obs)

        obs, reward, done, _ = env.step(action)
        total_reward += reward
        step_count += 1

        if render:
            env.render(action)
            student_agent.print_internal_state()
            state = student_agent.get_state(obs)
            print(f"{state=}")
            print(student_agent.q_table.get(state, None))
            # print(student_agent.update_count.get(state, None))
            time.sleep(0.15)

    print(f"Agent Finished in {step_count} steps, Score: {total_reward}")
    return total_reward


total_reward = 0
for _ in range(50):
    grid_size = random.randint(5, 10)
    reward = run_agent("student_agent.py", grid_size, True)
    total_reward += reward

print(f"Average reward: {total_reward / 50}")
