"""
Basic Reinforcement Learning example using Q-learning.

This script shows the core idea of reinforcement learning:
- an agent interacts with an environment
- it takes actions
- it receives rewards
- it learns which actions are better over time

This is a very simple example using a small 3x3 grid.
The goal is to reach the goal state and avoid the penalty state.
"""

import numpy as np

# ------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------
# Grid:
# 0 0 0
# 0 X 0
# 0 G 0
#
# Here:
# - X = trap/penalty state
# - G = goal state
# - 0 = normal states
#
# We use a 3x3 environment with states 0..8.
# The agent starts at state 0 and tries to reach goal state 8.

n_states = 9
n_actions = 4

Q = np.zeros((n_states, n_actions))

rewards = np.array([
    0, 0, 0,
    0, -10, 0,
    0, 0, 10
])
terminal_states = {4, 8}

def next_state(state, action):
    row, col = divmod(state, 3)

    if action == 0:  
        new_row = max(0, row - 1)
        new_col = col
    elif action == 1:  
        new_row = row
        new_col = min(2, col + 1)
    elif action == 2:  
        new_row = min(2, row + 1)
        new_col = col
    else:  
        new_row = row
        new_col = max(0, col - 1)

    return new_row * 3 + new_col

alpha = 0.8   
gamma = 0.9   
epsilon = 0.2 
num_episodes = 200

for episode in range(num_episodes):
    state = 0  

    while state not in terminal_states:
        if np.random.rand() < epsilon:
            action = np.random.choice(n_actions)
        else:
            action = np.argmax(Q[state])

        next_s = next_state(state, action)
        reward = rewards[next_s]

        best_next_value = np.max(Q[next_s])
        Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * best_next_value)

        state = next_s

print("Q-table after training:")
print(Q)

print("Best action from each state:")
for s in range(n_states):
    best_action = np.argmax(Q[s])
    print(f"State {s} -> action {best_action}")

for start in [0, 1, 2, 3]:
    state = start
    route = [state]

    while state not in terminal_states:
        action = np.argmax(Q[state])
        state = next_state(state, action)
        route.append(state)

    print(f"From state {start}, path: {route}")

