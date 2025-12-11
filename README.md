# Snake AI Competition

This project is a custom-built environment for testing and competing AI models in the classic Snake game. It is designed to run multiple game instances concurrently to visualize and compare different AI strategies.

## Features

- **Multi-Agent Environment**: Runs 3 concurrent Snake games in a single 900x600 window.
- **Custom Implementation**: Built from scratch using Python, `pygame`, and `numpy`.
- **High Performance**: Optimized to run smoothly at 90 FPS.
- **Extensible**: Designed to easily plug in different AI models.

## Competitors (The Models)

We will implement and compare three distinct AI approaches, each grounded in different mathematical frameworks.

### 1. Hamiltonian Cycle (Graph Theory)

This model treats the game grid as a graph $G = (V, E)$ where $V$ is the set of vertices (grid cells) and $E$ is the set of edges connecting adjacent cells. The goal is to find a **Hamiltonian Cycle**, a path that visits every vertex exactly once and returns to the start.

**Mathematical Formulation:**

Let $A$ be the adjacency matrix of $G$ such that $A_{ij} = 1$ if $(i, j) \in E$ and $0$ otherwise. A Hamiltonian path $P = (v_1, v_2, ..., v_N)$ where $N = |V|$ satisfies:

$$
\forall k \in \{1, ..., N-1\}, (v_k, v_{k+1}) \in E \quad \text{and} \quad (v_N, v_1) \in E
$$

$$
\bigcup_{k=1}^N \{v_k\} = V \quad \text{and} \quad v_i \neq v_j \text{ for } i \neq j
$$
\pi^*(s) = \arg\max_{a \in A} Q(s, a)
$$

### 3. Neuroevolution (Evolutionary Computation)

This model combines **Neural Networks** with **Genetic Algorithms**. Instead of backpropagation, we evolve the weights of the network over generations to maximize a fitness function (score).

**Mathematical Formulation:**

**Neural Network:**
Let $x$ be the input vector (state). The network output $y$ (action probabilities) for a single hidden layer with activation function $\phi$ (e.g., ReLU, Sigmoid) is given by:

$$
h = \phi(W_1 x + b_1)
$$
$$
y = \text{softmax}(W_2 h + b_2)
$$

Where $W_l$ and $b_l$ are the weight matrices and bias vectors for layer $l$.

**Genetic Algorithm:**
Let $\theta = \{W_1, b_1, W_2, b_2\}$ be the genome of an individual. We maintain a population $P = \{\theta_1, ..., \theta_M\}$.
The fitness function $F(\theta_i)$ evaluates the performance (e.g., game score).

The next generation $P_{t+1}$ is created through:
1.  **Selection**: Choosing parents based on fitness probability $p_i = \frac{F(\theta_i)}{\sum_j F(\theta_j)}$.
2.  **Crossover**: Combining genomes of parents $\theta_A, \theta_B$:
    $$ \theta_{child} = \beta \cdot \theta_A + (1 - \beta) \cdot \theta_B, \quad \beta \sim U(0, 1) $$
3.  **Mutation**: Adding random noise to weights with probability $\mu$:
    $$ \theta'_{j} = \theta_{j} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2) $$

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Mathweuzz/Snake-IA.git
   cd Snake-IA
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the multi-agent environment:

```bash
python main.py
```
