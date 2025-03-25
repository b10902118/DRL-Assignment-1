# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym

N_ACTIONS = 6
SIGHT = 5

MOVESOUTH = 0
MOVENORTH = 1
MOVEEAST = 2
MOVEWEST = 3
PICKUP = 4
DROPOFF = 5

try:
    with open("q_table.pkl", "rb") as f:
        q_table = pickle.load(f)
except:
    q_table = {}

initialized = False
has_passenger = False
passenger_possible_positions = []
destination_possible_positions = []

taxi_pos = None  # for convenience not to pass it around


def to_rel_pos(origin, target):
    return (target[0] - origin[0], target[1] - origin[1])


def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def init_internal_state(all_stations):
    global initialized, has_passenger, prev_all_stations, passenger_possible_positions, destination_possible_positions
    has_passenger = False
    passenger_possible_positions = all_stations
    destination_possible_positions = all_stations
    prev_all_stations = all_stations
    initialized = True


def print_internal_state():
    print(f"{has_passenger=}")
    print(f"{passenger_possible_positions=}")
    print(f"{destination_possible_positions=}")


def get_possible_positions(taxi_pos, possible_positions, target_look):
    if target_look:
        near_positions = [
            pos for pos in possible_positions if manhattan_distance(taxi_pos, pos) <= 1
        ]
        assert len(near_positions) > 0
        return near_positions
    else:
        far_positions = [
            pos for pos in possible_positions if manhattan_distance(taxi_pos, pos) > 1
        ]
        assert len(far_positions) > 0
        return far_positions


def elim_destination_positions(taxi_pos, destination_look):
    global destination_possible_positions
    destination_possible_positions = get_possible_positions(
        taxi_pos, destination_possible_positions, destination_look
    )


def elim_passenger_positions(taxi_pos, passenger_look):
    global passenger_possible_positions
    passenger_possible_positions = get_possible_positions(
        taxi_pos, passenger_possible_positions, passenger_look
    )


def parse_obs(obs):
    (
        taxi_row,
        taxi_col,
        s1_x,
        s1_y,
        s2_x,
        s2_y,
        s3_x,
        s3_y,
        s4_x,
        s4_y,
        obstacle_north,
        obstacle_south,
        obstacle_east,
        obstacle_west,
        passenger_look,
        destination_look,
    ) = obs
    taxi_pos = (taxi_row, taxi_col)
    all_stations = [(s1_x, s1_y), (s2_x, s2_y), (s3_x, s3_y), (s4_x, s4_y)]
    obstacles = (obstacle_south, obstacle_north, obstacle_east, obstacle_west)
    # obstacles = (obstacle_north, obstacle_south, obstacle_east, obstacle_west)
    return all_stations, taxi_pos, passenger_look, destination_look, obstacles


def set_taxi_pos(pos):
    global taxi_pos
    taxi_pos = pos


def clip_abs(n, max_n):
    if n < 0:
        return max(n, -max_n)
    else:
        return min(n, max_n)


def clip_relative_pos(pos, max_longer=SIGHT, max_shorter=SIGHT):
    # abs_pos = (abs(pos[0]), abs(pos[1]))
    return (clip_abs(pos[0], max_longer), clip_abs(pos[1], max_shorter))

    if abs_pos[0] > abs_pos[1]:
        return (clip_abs(pos[0], max_longer), clip_abs(pos[1], max_shorter))
    elif abs_pos[0] < abs_pos[1]:
        return (clip_abs(pos[0], max_shorter), clip_abs(pos[1], max_longer))
    else:
        return (clip_abs(pos[0], max_shorter), clip_abs(pos[1], max_shorter))


def get_state(obs):
    all_stations, taxi_pos, passenger_look, destination_look, obstacles = parse_obs(obs)

    set_taxi_pos(taxi_pos)

    # update internal state
    if not initialized:  # new round
        init_internal_state(all_stations)

    if (
        len(passenger_possible_positions) == 1
        and passenger_possible_positions == destination_possible_positions
        and not has_passenger
    ):
        return "Done"

    if has_passenger:
        elim_destination_positions(taxi_pos, destination_look)
    else:
        elim_destination_positions(taxi_pos, destination_look)
        elim_passenger_positions(taxi_pos, passenger_look)

    # print(f"{has_passenger=}")
    # print(f"{passenger_look=}")
    # print(f"{taxi_pos=}")
    # print(f"{passenger_possible_positions=}")
    # print(f"{destination_possible_positions=}")
    # print("=====================================")

    # eliminate possible stations and select the closest station
    possible_stations = (
        destination_possible_positions
        if has_passenger
        else passenger_possible_positions
    )

    closest_station_pos = min(
        possible_stations, key=lambda s: manhattan_distance(taxi_pos, s)
    )
    closest_possible_station_relative_pos = to_rel_pos(taxi_pos, closest_station_pos)
    closest_possible_station_relative_pos = clip_relative_pos(
        closest_possible_station_relative_pos
    )

    return (
        has_passenger,
        closest_possible_station_relative_pos,
        obstacles,
        passenger_look if not has_passenger else destination_look,
    )


# TODO: record passenger or destination when found. Add passenger station to empty after pickup
# TODO: support drop passenger non-destination by recording at drop


def post_update_internal_state(state, action):
    # prepare state for the next step
    global has_passenger, passenger_possible_positions
    has_passenger, closest_possible_station_relative_pos, obstacles, near_target = state
    if (
        action == PICKUP
        and not has_passenger
        and near_target
        and closest_possible_station_relative_pos == (0, 0)
    ):
        has_passenger = True

    elif action == DROPOFF and has_passenger:
        has_passenger = False
        passenger_possible_positions = [taxi_pos]


cnt = 0


def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Numeric stability
    return exp_x / exp_x.sum()


# TODO: remove qtable when submitting
def get_action(obs):

    global cnt
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys.
    #       Otherwise, even if your agent performs well in training, it may fail during testing.

    state = get_state(obs)
    action = np.random.choice(N_ACTIONS, p=softmax(q_table[state]))

    ## softmax action selection
    # if state in q_table:
    #    q_values = q_table[state]
    #    exp_q_values = np.exp(q_values - np.max(q_values))  # for numerical stability
    #    probabilities = exp_q_values / np.sum(exp_q_values)
    #    action = np.random.choice(len(q_values), p=probabilities)
    # else:
    #    action = np.random.choice(N_ACTIONS)

    post_update_internal_state(state, action)
    # print(action, state, end=" ")

    return action
