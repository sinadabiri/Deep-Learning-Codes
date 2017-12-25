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
            Q_values.append(['Collect Reward'] * 4)
            continue
        # The initial V_star for all actions. I put -10 as initial values. Then, if an action is impossible to be taken will not
        # be selected. If I initialized it as zero, then impossible action might still compete with possible action with V_value
        # equal to zero.
        candidate_V_star = [-10, -10, -10, -10]
        for action in actions:
            i, j = s
            if action == 'up':
                i += actions[action]
                if (i, j) in all_state:
                    candidate_V_star[0] = 1 * (R[i, j] + gama * V_star_past[i, j])
                continue
            if action == 'down':
                i += actions[action]
                if (i, j) in all_state:
                    candidate_V_star[1] = 1 * (R[i, j] + gama * V_star_past[i, j])
                continue
            if action == 'right':
                j += actions[action]
                if (i, j) in all_state:
                    candidate_V_star[2] = 1 * (R[i, j] + gama * V_star_past[i, j])
                continue
            if action == 'left':
                j += actions[action]
                if (i, j) in all_state:
                    candidate_V_star[3] = 1 * (R[i, j] + gama * V_star_past[i, j])
        i, j = s
        V_star_current[i, j] = max(candidate_V_star)
        optimal_action_index = candidate_V_star.index(max(candidate_V_star))
        New_pi_star = list(actions.keys())[optimal_action_index]
        pi_star[str(s)] = New_pi_star
        Q_values.append(candidate_V_star)

    convergence = np.amax(abs(V_star_current - V_star_past))
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

# Q values for all (state, action) pairs based on the optimal policies found in value-iteration algorithm
Q_values = np.array(Q_values, dtype=object)
print('\n')
print('Q-values Matrix, where rows are the sequqence of states as ordered above and the columns are:[Up, Down, Right, Left}')
print('\n')
print(Q_values)







