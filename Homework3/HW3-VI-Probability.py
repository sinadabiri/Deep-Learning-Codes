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

pi_star = dict([(str(s), 'nothing') for s in all_state])  # the dictionary for pi_star corresponding to each state.
pi_star[str((0, 3))] = 'Exit the system'  # the optimal policy (action) for terminal state, which already defined.
pi_star[str((1, 3))] = 'Exit the system'  # the optimal policy (action) for terminal state. which already defined.
pi_star[str((2, 1))] = 'Bounce back'

while threshold < convergence:
    Q_values = []
    # the loop is not iterate ove the null and terminal states.
    for s in all_state:
        if s in Null_state:
            Q_values.append(['No Value'] * 4)
            continue
        if s in terminal_state:
            Q_values.append(['Collect Reward', 'Null', 'Collect Reward', 'Null'])
            continue
        candidate_V_star = [0, 0, 0, 0]   # the V values for four actions. The max of this list is the V_star
        for action in actions:
            i, j = s
            if action == 'up':
                a = i + actions[action]
                b = j
                if (a, b) in all_state:
                    candidate_V_star[0] = 0.8 * (R[a, b] + gama * V_star_past[a, b])

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
                    candidate_V_star[0] += 0.2/len(number_possible_actions) * (R[a, b] + gama * V_star_past[a, b])
                continue

            if action == 'down':
                a = i + actions[action]
                b = j
                if (a, b) in all_state:
                    candidate_V_star[1] = 0.8 * (R[a, b] + gama * V_star_past[a, b])
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
                    candidate_V_star[1] += 0.2 / len(number_possible_actions) * (R[a, b] + gama * V_star_past[a, b])
                continue

            if action == 'right':
                a = i
                b = j + actions[action]
                if (a, b) in all_state:
                    candidate_V_star[2] = 0.8 * (R[a, b] + gama * V_star_past[a, b])
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
                    candidate_V_star[2] += 0.2 / len(number_possible_actions) * (R[a, b] + gama * V_star_past[a, b])
                continue

            if action == 'left':
                a = i
                b = j + actions[action]
                if (a, b) in all_state:
                    candidate_V_star[2] = 0.8 * (R[a, b] + gama * V_star_past[a, b])
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
                    candidate_V_star[3] += 0.2 / len(number_possible_actions) * (R[a, b] + gama * V_star_past[a, b])

        i, j = s
        V_star_current[i, j] = max(candidate_V_star)  # update V_star for state s (i, j)
        optimal_action_index = candidate_V_star.index(max(candidate_V_star))
        New_pi_star = list(actions.keys())[optimal_action_index]
        pi_star[str(s)] = New_pi_star  # update the policy for state s (i, j)
        Q_values.append(candidate_V_star)

    convergence = np.amax(abs(V_star_current - V_star_past))   # find the new convergence value to check if V_values have converged.
    V_star_past = V_star_current.copy()

V_star_current[0, 3] = 'collect all reward'
V_star_current[1, 3] = 'collect all reward'
V_star_current[2, 1] = 'nothing'
for state in all_state:
    i, j = state
    x = i + 1
    y = j + 1
    print('State {}: Optimal Value Function is {}. '
          'Its Optimal Policy (Action) is {}'.format((x, y), V_star_current[i, j], pi_star[str(state)]))







