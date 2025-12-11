# Snake AI Competition

This project is a custom-built environment for testing and competing AI models in the classic Snake game. It is designed to run multiple game instances concurrently to visualize and compare different AI strategies.

## Features

- **Multi-Agent Environment**: Runs 3 concurrent Snake games in a single 900x600 window.
- **Custom Implementation**: Built from scratch using Python, `pygame`, and `numpy`.
- **High Performance**: Optimized to run smoothly at 90 FPS.
- **Extensible**: Designed to easily plug in different AI models (Random, Neural Networks, etc.).

## Current State

The project currently sets up the environment with 3 independent game loops. The agents are currently using a random movement strategy as a placeholder for future AI implementations.

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

## Structure

- `main.py`: Entry point. Manages the Pygame window and the 3 game instances.
- `game.py`: Contains the `SnakeGame` class, encapsulating the logic for a single game.
- `requirements.txt`: Project dependencies.
