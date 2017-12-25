import numpy as np
from collections import OrderedDict

actions = OrderedDict([('up', -1), ('down', 1), ('right', 1), ('left', -1)])
threshold = 0.01  # convergence threshold
convergence = 2 * threshold  # this is a value just to be allowed to enter the while loop.
gama = 0.9

# Reward matrix
R = np.zeros((3, 4))
R[0, 3] = 10
R[1, 3] = -10

# state list, shows all possible state (including null and terminal state)
all_state = []
for k in range(3):
    for z in range(4):
        all_state.append((k, z))
Null_state = [(2, 1)]  # Null state
terminal_state = [(0, 3), (1, 3)]  # Terminal states
remove_state = [(0, 3), (1, 3), (2, 1)]

V_star_past = np.zeros((3, 4))  # Initial V_star
V_star_current = np.zeros((3, 4), dtype=object)  # this matrix is going to be updated through iterations


pi_star_past = OrderedDict([(str(s), '') for s in all_state])  # the pi_star dict for the previous step
pi_star_current = OrderedDict([(str(s), 'up') for s in all_state])  # pi_star dict for the current step, which is updated iteratively
pi_star_current[str((0, 3))] = 'Exit the system'
pi_star_current[str((1, 3))] = 'Exit the system'
pi_star_current[str((2, 1))] = 'Bounce back'

pi_star_past[str((0, 3))] = 'Exit the system'
pi_star_past[str((1, 3))] = 'Exit the system'
pi_star_past[str((2, 1))] = 'Bounce back'
while pi_star_past != pi_star_current:
    pi_star_past = pi_star_current.copy()

    while threshold < convergence:
        # the loop is not iterate ove the null and terminal states.
        for s in all_state:
            if s in Null_state:
                continue
            if s in terminal_state:
                continue

            # Start to pass over possible states
            action = pi_star_current[str(s)]
            i, j = s
            if action == 'up':
                a = i + actions[action]
                b = j
                if (a, b) in all_state:
                    V_star_current[a, b] = 0.8 * (R[a, b] + gama * V_star_past[a, b])

                # Start to collect reward from other possible states with lower probability.
                remain_possible_action = []
                a = i + actions['down']
                b = j
                remain_possible_action.append((a, b))
                a = i
                b = j + actions['left']
                remain_possible_action.append((a, b))
                a = i
                b = j + actions['right']
                remain_possible_action.append((a, b))
                number_possible_actions = []
                number_possible_actions = [item for item in remain_possible_action if item in all_state]
                for item in number_possible_actions:
                    a, b = item
                    V_star_current[a, b] += 0.2/len(number_possible_actions) * (R[a, b] + gama * V_star_past[a, b])
                continue

            if action == 'down':
                a = i + actions[action]
                b = j
                if (a, b) in all_state:
                    V_star_current[a, b] = 0.8 * (R[a, b] + gama * V_star_past[a, b])

                # Start to collect reward from other possible states with lower probability.
                remain_possible_action = []
                a = i + actions['up']
                b = j
                remain_possible_action.append((a, b))
                a = i
                b = j + actions['right']
                remain_possible_action.append((a, b))
                a = i
                b = j + actions['left']
                remain_possible_action.append((a, b))
                number_possible_actions = []
                number_possible_actions = [item for item in remain_possible_action if item in all_state]
                for item in number_possible_actions:
                    a, b = item
                    V_star_current[a, b] += 0.2 / len(number_possible_actions) * (R[a, b] + gama * V_star_past[a, b])
                continue

            if action == 'right':
                a = i
                b = j + actions[action]
                if (a, b) in all_state:
                    V_star_current[a, b] = 0.8 * (R[a, b] + gama * V_star_past[a, b])
                # Start to collect reward from other possible states with lower probability.
                remain_possible_action = []
                a = i + actions['up']
                b = j
                remain_possible_action.append((a, b))
                a = i + actions['down']
                b = j
                remain_possible_action.append((a, b))
                a = i
                b = j + actions['left']
                remain_possible_action.append((a, b))
                number_possible_actions = []
                number_possible_actions = [item for item in remain_possible_action if item in all_state]
                for item in number_possible_actions:
                    a, b = item
                    V_star_current[a, b] += 0.2 / len(number_possible_actions) * (R[a, b] + gama * V_star_past[a, b])
                continue

            if action == 'left':
                a = i
                b = j + actions[action]
                if (a, b) in all_state:
                    V_star_current[a, b] = 0.8 * (R[a, b] + gama * V_star_past[a, b])
                # Start to collect reward from other possible states with lower probability.
                remain_possible_action = []
                a = i + actions['up']
                b = j
                remain_possible_action.append((a, b))
                a = i + actions['down']
                b = j
                remain_possible_action.append((a, b))
                a = i
                b = j + actions['right']
                remain_possible_action.append((a, b))
                number_possible_actions = []
                number_possible_actions = [item for item in remain_possible_action if item in all_state]
                for item in number_possible_actions:
                    a, b = item
                    V_star_current[a, b] += 0.2 / len(number_possible_actions) * (R[a, b] + gama * V_star_past[a, b])

        convergence = np.amax(abs(V_star_current - V_star_past))
        V_star_past = V_star_current.copy()

    for s in all_state:
        if s in Null_state:
            continue
        if s in terminal_state:
            continue
        candidate_V_star = [0, 0, 0, 0]

        # Iterate over all possible actions to find the optimal action (policy)
        for action in actions:
            i, j = s
            if action == 'up':
                a = i + actions[action]
                b = j
                # First collect reward from the intended state with highest probability.
                if (a, b) in all_state:
                    candidate_V_star[0] = 0.8 * (R[a, b] + gama * V_star_current[a, b])

                # Start to collect reward from other possible states with lower probability.
                remain_possible_action = []
                a = i + actions['down']
                b = j
                remain_possible_action.append((a, b))
                a = i
                b = j + actions['left']
                remain_possible_action.append((a, b))
                a = i
                b = j + actions['right']
                remain_possible_action.append((a, b))
                number_possible_actions = []
                number_possible_actions = [item for item in remain_possible_action if item in all_state]
                for item in number_possible_actions:
                    a, b = item
                    candidate_V_star[0] += 0.2/len(number_possible_actions) * (R[a, b] + gama * V_star_current[a, b])
                continue

            if action == 'down':
                a = i + actions[action]
                b = j
                if (a, b) in all_state:
                    candidate_V_star[1] = 0.8 * (R[a, b] + gama * V_star_current[a, b])
                # Start to collect reward from other possible states with lower probability.
                remain_possible_action = []
                a = i + actions['up']
                b = j
                remain_possible_action.append((a, b))
                a = i
                b = j + actions['right']
                remain_possible_action.append((a, b))
                a = i
                b = j + actions['left']
                remain_possible_action.append((a, b))
                number_possible_actions = []
                number_possible_actions = [item for item in remain_possible_action if item in all_state]
                for item in number_possible_actions:
                    a, b = item
                    candidate_V_star[1] += 0.2 / len(number_possible_actions) * (R[a, b] + gama * V_star_current[a, b])
                continue

            if action == 'right':
                a = i
                b = j + actions[action]
                if (a, b) in all_state:
                    candidate_V_star[2] = 0.8 * (R[a, b] + gama * V_star_current[a, b])
                # Start to collect reward from other possible states with lower probability.
                remain_possible_action = []
                a = i + actions['up']
                b = j
                remain_possible_action.append((a, b))
                a = i + actions['down']
                b = j
                remain_possible_action.append((a, b))
                a = i
                b = j + actions['left']
                remain_possible_action.append((a, b))
                number_possible_actions = []
                number_possible_actions = [item for item in remain_possible_action if item in all_state]
                for item in number_possible_actions:
                    a, b = item
                    candidate_V_star[2] += 0.2 / len(number_possible_actions) * (R[a, b] + gama * V_star_current[a, b])
                continue

            if action == 'left':
                a = i
                b = j + actions[action]
                if (a, b) in all_state:
                    candidate_V_star[2] = 0.8 * (R[a, b] + gama * V_star_current[a, b])
                # Start to collect reward from other possible states with lower probability.
                remain_possible_action = []
                a = i + actions['up']
                b = j
                remain_possible_action.append((a, b))
                a = i + actions['down']
                b = j
                remain_possible_action.append((a, b))
                a = i
                b = j + actions['right']
                remain_possible_action.append((a, b))
                number_possible_actions = []
                number_possible_actions = [item for item in remain_possible_action if item in all_state]
                for item in number_possible_actions:
                    a, b = item
                    candidate_V_star[3] += 0.2 / len(number_possible_actions) * (R[a, b] + gama * V_star_current[a, b])
        i, j = s
        optimal_action_index = candidate_V_star.index(max(candidate_V_star))
        New_pi_star = list(actions.keys())[optimal_action_index]
        pi_star_current[str(s)] = New_pi_star

V_star_current[0, 3] = 'collect all reward'
V_star_current[1, 3] = 'collect all reward'
V_star_current[2, 1] = 'nothing'
for state in all_state:
    i, j = state
    x = i + 1
    y = j + 1
    print('State {}: Optimal Value Function is {}. '
          'Its Optimal Policy (Action) is {}'.format((x, y), V_star_current[i, j], pi_star_current[str(state)]))








