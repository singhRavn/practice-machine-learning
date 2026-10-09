"""
One-step-advanced Reinforcement Learning example using Q-learning.

This version adds a more realistic training setup:
- a reward table for each state
- epsilon decay for exploration vs exploitation
- a training loop over many episodes
- a policy evaluation step to see what the agent learns

The goal is still the same: learn which actions lead to high rewards.
"""

import numpy as np

n_states = 9
n_actions = 4

# Q-table: state-action values
Q = np.zeros((n_states, n_actions))


rewards = np.array([
    0, 0, 0,
    0, -10, 0,
    0, 0, 10
])

terminal_states = {4, 8}


def next_state(state, action):
    """Return the next state after taking an action from a given state."""
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
epsilon = 1.0    
min_epsilon = 0.05
decay = 0.995
num_episodes = 500

for episode in range(num_episodes):
    state = 0

    while state not in terminal_states:
        # Explore sometimes, exploit learned knowledge otherwise
        if np.random.rand() < epsilon:
            action = np.random.choice(n_actions)
        else:
            action = np.argmax(Q[state])

        next_s = next_state(state, action)
        reward = rewards[next_s]

        # Q-learning update
        best_next_value = np.max(Q[next_s])
        Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * best_next_value)

        state = next_s

    # Reduce exploration over time
    epsilon = max(min_epsilon, epsilon * decay)

print("Q-table after training:")
print(Q)

print("\nBest action from each state:")
for s in range(n_states):
    best_action = np.argmax(Q[s])
    print(f"State {s} -> action {best_action}")

# Follow the learned policy from several starting states
print("\nPolicy rollout from sample start states:")
for start in [0, 1, 2, 3]:
    state = start
    route = [state]

    while state not in terminal_states:
        action = np.argmax(Q[state])
        state = next_state(state, action)
        route.append(state)

    print(f"Start {start}: {route}")


