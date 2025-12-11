import numpy as np
import random

class QLearningAgent:
    def __init__(self, n_actions=4, lr=0.01, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}

    def get_q_values(self, state):
        state_key = tuple(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions)
        return self.q_table[state_key]

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        
        q_values = self.get_q_values(state)
        return np.argmax(q_values)

    def train(self, state, action, reward, next_state, done):
        state_key = tuple(state)
        next_state_key = tuple(next_state)
        
        current_q = self.get_q_values(state)[action]
        
        if done:
            target_q = reward
        else:
            next_max_q = np.max(self.get_q_values(next_state))
            target_q = reward + self.gamma * next_max_q
            
        # Bellman Equation
        new_q = (1 - self.lr) * current_q + self.lr * target_q
        self.q_table[state_key][action] = new_q

        if done and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
