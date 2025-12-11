import numpy as np
import random

class ActorCritic:
    def __init__(self, input_size, hidden_size, output_size, lr=0.001):
        self.lr = lr
        
        # Shared or separate? Let's use separate for simplicity in gradients
        # Actor (Policy)
        self.W1_a = np.random.randn(input_size, hidden_size) * np.sqrt(2.0/input_size)
        self.b1_a = np.zeros(hidden_size)
        self.W2_a = np.random.randn(hidden_size, output_size) * np.sqrt(2.0/hidden_size)
        self.b2_a = np.zeros(output_size)
        
        # Critic (Value)
        self.W1_c = np.random.randn(input_size, hidden_size) * np.sqrt(2.0/input_size)
        self.b1_c = np.zeros(hidden_size)
        self.W2_c = np.random.randn(hidden_size, 1) * np.sqrt(2.0/hidden_size)
        self.b2_c = np.zeros(1)

    def forward(self, x):
        # Actor
        z1_a = np.dot(x, self.W1_a) + self.b1_a
        a1_a = np.maximum(0, z1_a)
        z2_a = np.dot(a1_a, self.W2_a) + self.b2_a
        
        # Softmax
        exp_logits = np.exp(z2_a - np.max(z2_a, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Critic
        z1_c = np.dot(x, self.W1_c) + self.b1_c
        a1_c = np.maximum(0, z1_c)
        value = np.dot(a1_c, self.W2_c) + self.b2_c
        
        return probs, value

    def get_action(self, state):
        state = state.reshape(1, -1)
        probs, _ = self.forward(state)
        action = np.random.choice(len(probs[0]), p=probs[0])
        return action, probs[0][action]

    def update(self, states, actions, old_log_probs, returns, advantages, clip_param=0.2):
        # Forward pass
        probs, values = self.forward(states)
        values = values.flatten()
        
        # Gather probs for actions taken
        # actions is (batch,) indices
        batch_size = len(actions)
        action_probs = probs[np.arange(batch_size), actions]
        
        # Ratio = pi / old_pi
        # Use log probs for stability? log(a/b) = log(a) - log(b) -> exp(log_a - log_b)
        # old_log_probs passed in are actually just probs? Let's assume passed as probs for simplicity first
        # Actually, let's use standard ratio
        
        ratio = action_probs / (old_log_probs + 1e-10)
        
        # Surrogate Loss 1
        surr1 = ratio * advantages
        
        # Surrogate Loss 2 (Clipped)
        surr2 = np.clip(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
        
        # Policy Loss (Maximize -> Minimize negative)
        policy_loss = -np.mean(np.minimum(surr1, surr2))
        
        # Value Loss
        value_loss = np.mean((returns - values)**2)
        
        # Entropy Bonus (Optional, for exploration)
        entropy = -np.mean(np.sum(probs * np.log(probs + 1e-10), axis=1))
        entropy_coef = 0.01
        
        total_loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy
        
        # Gradients (Simplified: Re-computing for backprop manually is hard for PPO in one go)
        # We will do a simplified backprop for Actor and Critic separately based on their losses
        
        # Critic Backprop (MSE)
        d_val = -2 * (returns - values) / batch_size # dL/dV
        d_val = d_val.reshape(-1, 1)
        
        # Critic Hidden
        z1_c = np.dot(states, self.W1_c) + self.b1_c
        a1_c = np.maximum(0, z1_c)
        
        dW2_c = np.dot(a1_c.T, d_val)
        db2_c = np.sum(d_val, axis=0)
        
        da1_c = np.dot(d_val, self.W2_c.T)
        dz1_c = da1_c * (z1_c > 0)
        
        dW1_c = np.dot(states.T, dz1_c)
        db1_c = np.sum(dz1_c, axis=0)
        
        # Actor Backprop (Policy Gradient)
        # dL/dProb = - (advantage / old_prob) if unclipped... clipped is harder.
        # Let's approximate gradient using the clipped objective derivative
        # If ratio inside clip range: grad = advantage / old_prob
        # If outside: grad = 0
        
        d_probs = np.zeros_like(probs)
        for i in range(batch_size):
            r = ratio[i]
            adv = advantages[i]
            a = actions[i]
            old_p = old_log_probs[i]
            
            grad = 0
            if 1.0 - clip_param <= r <= 1.0 + clip_param:
                grad = -adv / (old_p + 1e-10)
            
            # Add entropy grad: -coef * (1 + log(p))
            grad += -entropy_coef * (1 + np.log(probs[i][a] + 1e-10))
            
            d_probs[i][a] = grad
            
        # Backprop through Softmax
        # dL/dz = p_i - y_i if CrossEntropy... here we have arbitrary dL/dp
        # Softmax Jacobian is complex.
        # Standard trick: dL/dz_i = p_i * (dL/dp_i - sum(p_k * dL/dp_k))
        
        d_logits = np.zeros_like(probs)
        for i in range(batch_size):
            d_p = d_probs[i]
            p = probs[i]
            sum_p_dp = np.sum(p * d_p)
            d_logits[i] = p * (d_p - sum_p_dp)
            
        z1_a = np.dot(states, self.W1_a) + self.b1_a
        a1_a = np.maximum(0, z1_a)
        
        dW2_a = np.dot(a1_a.T, d_logits)
        db2_a = np.sum(d_logits, axis=0)
        
        da1_a = np.dot(d_logits, self.W2_a.T)
        dz1_a = da1_a * (z1_a > 0)
        
        dW1_a = np.dot(states.T, dz1_a)
        db1_a = np.sum(dz1_a, axis=0)
        
        # Update
        self.W1_c -= self.lr * dW1_c
        self.b1_c -= self.lr * db1_c
        self.W2_c -= self.lr * dW2_c
        self.b2_c -= self.lr * db2_c
        
        self.W1_a -= self.lr * dW1_a
        self.b1_a -= self.lr * db1_a
        self.W2_a -= self.lr * dW2_a
        self.b2_a -= self.lr * db2_a
        
        return total_loss

class PPOAgent:
    def __init__(self, input_size=24, hidden_size=24, output_size=4, lr=0.001, gamma=0.99, gae_lambda=0.95, clip_param=0.2):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        
        self.model = ActorCritic(input_size, hidden_size, output_size, lr)
        
        self.states = []
        self.actions = []
        self.probs = []
        self.rewards = []
        self.dones = []
        self.batch_size = 64 # Steps to collect before update
        
        # Load weights if exist
        self.load_model()

    def get_action(self, state):
        action, prob = self.model.get_action(state)
        return action, prob

    def store_transition(self, state, action, prob, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.rewards.append(reward)
        self.dones.append(done)

    def train(self):
        if len(self.states) < self.batch_size:
            return

        states = np.array(self.states)
        actions = np.array(self.actions)
        old_probs = np.array(self.probs)
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        
        # Compute GAE
        _, values = self.model.forward(states)
        values = values.flatten()
        
        # Append 0 for last next_value (simplified)
        values = np.append(values, 0)
        
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae_lam = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = 0 # Simplified
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t+1]
                
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            
        returns = advantages + values[:-1]
        
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)
        
        # Update Epochs
        for _ in range(4): # 4 epochs
            self.model.update(states, actions, old_probs, returns, advantages, self.clip_param)
            
        # Clear memory
        self.states = []
        self.actions = []
        self.probs = []
        self.rewards = []
        self.dones = []

    def save_model(self, filename="models/ppo_weights.npy"):
        weights = [
            self.model.W1_a, self.model.b1_a, self.model.W2_a, self.model.b2_a,
            self.model.W1_c, self.model.b1_c, self.model.W2_c, self.model.b2_c
        ]
        np.save(filename, np.array(weights, dtype=object))
        print("PPO model saved.")

    def load_model(self, filename="models/ppo_weights.npy"):
        try:
            weights = np.load(filename, allow_pickle=True)
            self.model.W1_a, self.model.b1_a, self.model.W2_a, self.model.b2_a = weights[0], weights[1], weights[2], weights[3]
            self.model.W1_c, self.model.b1_c, self.model.W2_c, self.model.b2_c = weights[4], weights[5], weights[6], weights[7]
            print("PPO model loaded successfully.")
        except FileNotFoundError:
            print("No saved PPO model found. Starting from scratch.")
