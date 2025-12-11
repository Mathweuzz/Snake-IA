import numpy as np
import random
from collections import deque

class DuelingDQN:
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr
        
        # He initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0/input_size)
        self.b1 = np.zeros(hidden_size)
        
        # Value Stream (Hidden -> 1)
        self.W_val = np.random.randn(hidden_size, 1) * np.sqrt(2.0/hidden_size)
        self.b_val = np.zeros(1)
        
        # Advantage Stream (Hidden -> Output)
        self.W_adv = np.random.randn(hidden_size, output_size) * np.sqrt(2.0/hidden_size)
        self.b_adv = np.zeros(output_size)

    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1) # ReLU
        
        # Value
        self.val = np.dot(self.a1, self.W_val) + self.b_val
        
        # Advantage
        self.adv = np.dot(self.a1, self.W_adv) + self.b_adv
        
        # Aggregation: Q = V + (A - mean(A))
        q_values = self.val + (self.adv - np.mean(self.adv, axis=1, keepdims=True))
        return q_values

    def backward(self, x, y_target):
        # Forward pass
        y_pred = self.forward(x)
        
        # Gradient of MSE Loss
        dy_pred = (y_pred - y_target) / x.shape[0]
        
        # Gradients for Aggregation Layer
        # dQ/dV = 1
        # dQ/dA = 1 - 1/N
        
        d_val = np.sum(dy_pred, axis=1, keepdims=True)
        d_adv = dy_pred - np.mean(dy_pred, axis=1, keepdims=True)
        
        # Value Stream Gradients
        dW_val = np.dot(self.a1.T, d_val)
        db_val = np.sum(d_val, axis=0)
        
        # Advantage Stream Gradients
        dW_adv = np.dot(self.a1.T, d_adv)
        db_adv = np.sum(d_adv, axis=0)
        
        # Hidden Layer Gradients
        da1_val = np.dot(d_val, self.W_val.T)
        da1_adv = np.dot(d_adv, self.W_adv.T)
        da1 = da1_val + da1_adv
        
        dz1 = da1 * (self.z1 > 0)
        
        dW1 = np.dot(x.T, dz1)
        db1 = np.sum(dz1, axis=0)
        
        # Update weights
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W_val -= self.lr * dW_val
        self.b_val -= self.lr * db_val
        self.W_adv -= self.lr * dW_adv
        self.b_adv -= self.lr * db_adv
        
        loss = np.mean((y_pred - y_target)**2)
        return loss
        
    def get_weights(self):
        return [self.W1, self.b1, self.W_val, self.b_val, self.W_adv, self.b_adv]
        
    def set_weights(self, weights):
        self.W1, self.b1, self.W_val, self.b_val, self.W_adv, self.b_adv = weights

class QLearningAgent:
    def __init__(self, n_actions=4, input_size=24, hidden_size=24, lr=0.001, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = lr
        
        self.model = DuelingDQN(input_size, hidden_size, n_actions, lr)
        self.target_model = DuelingDQN(input_size, hidden_size, n_actions, lr)
        self.update_target_network()
        
        self.memory = deque(maxlen=2000)
        self.batch_size = 32
        
        # Load weights if exist
        self.load_model()

    def update_target_network(self):
        self.target_model.set_weights([w.copy() for w in self.model.get_weights()])

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randint(0, self.n_actions - 1)
        
        state = state.reshape(1, -1)
        q_values = self.model.forward(state)
        return np.argmax(q_values[0])

    def train(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])
        
        # Double DQN:
        # 1. Select action using Online Model
        next_q_online = self.model.forward(next_states)
        best_actions = np.argmax(next_q_online, axis=1)
        
        # 2. Evaluate action using Target Model
        next_q_target = self.target_model.forward(next_states)
        
        # 3. Compute Target Q
        target_q = self.model.forward(states) # Current predictions
        
        for i in range(self.batch_size):
            target = rewards[i]
            if not dones[i]:
                # Q(s,a) = r + gamma * Q_target(s', argmax Q_online(s', a'))
                target += self.gamma * next_q_target[i][best_actions[i]]
            target_q[i][actions[i]] = target
            
        loss = self.model.backward(states, target_q)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # Soft update target network? Or hard update every N steps?
        # Let's do a very slow soft update every step for simplicity
        tau = 0.01
        online_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        new_weights = []
        for o, t in zip(online_weights, target_weights):
            new_weights.append(tau * o + (1 - tau) * t)
        self.target_model.set_weights(new_weights)

    def save_model(self, filename="models/dqn_weights.npy"):
        np.save(filename, np.array(self.model.get_weights(), dtype=object))
        # Also save epsilon?
        # For now just weights.

    def load_model(self, filename="models/dqn_weights.npy"):
        try:
            weights = np.load(filename, allow_pickle=True)
            self.model.set_weights(weights)
            self.update_target_network()
            print("Model loaded successfully.")
            # If model loaded, lower epsilon to exploit more?
            self.epsilon = 0.5 # Start with some exploration but not 1.0
        except FileNotFoundError:
            print("No saved model found. Starting from scratch.")
