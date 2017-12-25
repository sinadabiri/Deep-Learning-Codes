import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
a = [0, 0, -1]
b = a.index(0)

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

pi_star_past = OrderedDict([(str(s), '') for s in all_state])  # the pi_star dict for the previous step
pi_star_current = OrderedDict([(str(s), 'up') for s in all_state])  # pi_star dict for the current step, which is updated iteratively
pi_star_current[str((0, 3))] = 'Exit the system'
pi_star_current[str((1, 3))] = 'Exit the system'
pi_star_current[str((2, 1))] = 'Bounce back'

pi_star_past[str((0, 3))] = 'Exit the system'
pi_star_past[str((1, 3))] = 'Exit the system'
pi_star_past[str((2, 1))] = 'Bounce back'

# Quality policy at each time
quality_policy = [0]
while pi_star_past != pi_star_current:
    pi_star_past = pi_star_current.copy()

    while threshold < convergence:
        for s in all_state:
            if s in Null_state:
                continue
            if s in terminal_state:
                continue

            #Start to pass over possible states
            action = pi_star_current[str(s)]
            i, j = s
            if action == 'up':
                i += actions[action]
                if (i, j) in all_state:
                    V_star_current[i, j] = 1 * (R[i, j] + gama * V_star_past[i, j])
                continue
            if action == 'down':
                i += actions[action]
                if (i, j) in all_state:
                    V_star_current[i, j] = 1 * (R[i, j] + gama * V_star_past[i, j])
                continue
            if action == 'right':
                j += actions[action]
                if (i, j) in all_state:
                    V_star_current[i, j] = 1 * (R[i, j] + gama * V_star_past[i, j])
                continue
            if action == 'left':
                j += actions[action]
                if (i, j) in all_state:
                    V_star_current[i, j] = 1 * (R[i, j] + gama * V_star_past[i, j])

        convergence = np.amax(abs(V_star_current - V_star_past))
        V_star_past = V_star_current.copy()

    # Collect the quality policy at each time
    quality_policy.append(np.sum(V_star_current))
    for s in all_state:
        if s in Null_state:
            continue
        if s in terminal_state:
            continue
        candidate_V_star = [-10, -10, -10, -10]  # For all actions in a given state.-10 means impossible to take that action
        for action in actions:
            i, j = s
            if action == 'up':
                i += actions[action]
                if (i, j) in all_state:
                    candidate_V_star[0] = 1 * (R[i, j] + gama * V_star_current[i, j])
                continue
            if action == 'down':
                i += actions[action]
                if (i, j) in all_state:
                    candidate_V_star[1] = 1 * (R[i, j] + gama * V_star_current[i, j])
                continue
            if action == 'right':
                j += actions[action]
                if (i, j) in all_state:
                    candidate_V_star[2] = 1 * (R[i, j] + gama * V_star_current[i, j])
                continue
            if action == 'left':
                j += actions[action]
                if (i, j) in all_state:
                    candidate_V_star[3] = 1 * (R[i, j] + gama * V_star_current[i, j])
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

plt.figure(1)

x = np.linspace(0, len(quality_policy) - 1, len(quality_policy))
plt.plot(x, quality_policy, 'b')
plt.ylabel("sum of v_values for all states")
plt.xlabel("Iteration when policies are updated")
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.title('Quality Policy')
plt.show()