import numpy as np
import random

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros(output_size)

    def forward(self, x):
        # Input layer -> Hidden layer
        z1 = np.dot(x, self.W1) + self.b1
        a1 = np.maximum(0, z1) # ReLU
        
        # Hidden layer -> Output layer
        z2 = np.dot(a1, self.W2) + self.b2
        
        # Softmax
        exp_scores = np.exp(z2 - np.max(z2)) # Stability
        probs = exp_scores / np.sum(exp_scores)
        
        return probs

    def get_weights(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def set_weights(self, weights):
        self.W1, self.b1, self.W2, self.b2 = weights

class NeuroEvolutionAgent:
    def __init__(self, input_size, hidden_size, output_size, population_size=50, mutation_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        
        self.population = [NeuralNetwork(input_size, hidden_size, output_size) for _ in range(population_size)]
        self.fitness_scores = [0] * population_size
        self.current_individual_idx = 0
        self.generation = 1
        self.fitness_history = []

    def get_action(self, state):
        nn = self.population[self.current_individual_idx]
        probs = nn.forward(state)
        return np.argmax(probs)

    def update_fitness(self, score):
        self.fitness_scores[self.current_individual_idx] = score

    def next_individual(self):
        self.current_individual_idx += 1
        if self.current_individual_idx >= self.population_size:
            self.evolve()
            self.current_individual_idx = 0
            self.generation += 1

    def evolve(self):
        # Selection: Sort by fitness
        sorted_indices = np.argsort(self.fitness_scores)[::-1]
        
        # Elitism: Keep top 10%
        elite_count = int(self.population_size * 0.1)
        new_population = []
        
        for i in range(elite_count):
            idx = sorted_indices[i]
            elite_nn = self.population[idx]
            # Clone
            new_nn = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
            new_nn.set_weights([w.copy() for w in elite_nn.get_weights()])
            new_population.append(new_nn)
            
        # Crossover & Mutation for the rest
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            child_weights = self.crossover(parent1, parent2)
            child_weights = self.mutate(child_weights)
            
            child_nn = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
            child_nn.set_weights(child_weights)
            new_population.append(child_nn)
            
        self.population = new_population
        best_fitness = self.fitness_scores[sorted_indices[0]] if self.fitness_scores else 0
        self.fitness_history.append(best_fitness)
        self.fitness_scores = [0] * self.population_size
        print(f"Generation {self.generation} complete. Best fitness: {best_fitness}")

    def tournament_selection(self, k=3):
        indices = np.random.choice(self.population_size, k)
        best_idx = indices[0]
        for idx in indices:
            if self.fitness_scores[idx] > self.fitness_scores[best_idx]:
                best_idx = idx
        return self.population[best_idx]

    def crossover(self, p1, p2):
        w1 = p1.get_weights()
        w2 = p2.get_weights()
        child_w = []
        
        for i in range(len(w1)):
            # Random crossover point or uniform crossover? Let's do uniform
            mask = np.random.rand(*w1[i].shape) > 0.5
            c_w = np.where(mask, w1[i], w2[i])
            child_w.append(c_w)
            
        return child_w

    def mutate(self, weights):
        mutated_w = []
        for w in weights:
            mask = np.random.rand(*w.shape) < self.mutation_rate
            noise = np.random.randn(*w.shape) * 0.5 # Mutation strength
            m_w = w + mask * noise
            mutated_w.append(m_w)
        return mutated_w
