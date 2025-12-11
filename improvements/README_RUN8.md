# Improvement Report - Run 8

## Benchmark Configuration
- **Duration**: 2 minutes
- **Date**: 2025-12-11

## Results Analysis

### 1. PPO
- **Score**: 0 (Final), 1 (Peak)
- **Deaths**: 17
- **Analysis**: PPO scored once! It's learning. However, it starts from scratch every run.

### 2. D3QN
- **Best Score**: 1
- **Deaths**: 13
- **Score History**: [0, ..., 1, 0, 0, 1, 0]
- **Analysis**: D3QN scored twice. Persistence is working; it's building knowledge.

### 3. Neuroevolution
- **Generations**: 3
- **Analysis**: Fitness is low (all 1s). The snakes are dying before eating.

## Action Plan
1.  **PPO Persistence**: Save/Load PPO weights like D3QN to enable cumulative learning.
2.  **Neuroevolution Survival Bonus**: Add a time-survived component to fitness to reward longer games.
3.  **Faster Game Speed**: Reduce `move_interval` to allow more training steps per minute.
