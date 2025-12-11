import pygame
import random
from game import SnakeGame

pygame.init()

# Display dimensions
TOTAL_WIDTH = 900
HEIGHT = 600
GAME_WIDTH = 300

display = pygame.display.set_mode((TOTAL_WIDTH, HEIGHT))
pygame.display.set_caption('Snake AI Competition')

clock = pygame.time.Clock()

from models.ppo import PPOAgent
from models.qlearning import QLearningAgent
from models.neuroevolution import NeuroEvolutionAgent

# Initialize 3 games
games = [
    SnakeGame(GAME_WIDTH, HEIGHT, 0, "PPO (Proximal Policy Opt)"),
    SnakeGame(GAME_WIDTH, HEIGHT, 300, "Q-Learning (D3QN)"),
    SnakeGame(GAME_WIDTH, HEIGHT, 600, "Neuroevolution")
]

# Initialize Agents
# Grid size is 300x600, block size 10 -> 30x60 grid
ppo_agent = PPOAgent(input_size=24, hidden_size=24, output_size=4)
q_agent = QLearningAgent(n_actions=4, input_size=24, hidden_size=24)
# Input: 24 (Vision), Hidden: 24, Output: 4 actions
neuro_agent = NeuroEvolutionAgent(input_size=24, hidden_size=24, output_size=4, population_size=12)

import json
import time

def main():
    start_time = time.time()
    BENCHMARK_DURATION = 120 # 2 minutes
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        display.fill((0, 0, 0))
        
        # Calculate stats
        elapsed_time = int(time.time() - start_time)
        time_str = f"{elapsed_time // 60:02d}:{elapsed_time % 60:02d}"
        
        # Benchmark Check
        if elapsed_time >= BENCHMARK_DURATION:
            print("Benchmark complete. Generating report...")
            report = {
                "PPO": {
                    "score": games[0].score,
                    "deaths": games[0].deaths,
                    "score_history": games[0].score_history
                },
                "Q-Learning": {
                    "best_score": games[1].best_score,
                    "deaths": games[1].deaths,
                    "score_history": games[1].score_history,
                    "final_epsilon": q_agent.epsilon
                },
                "Neuroevolution": {
                    "generations": neuro_agent.generation,
                    "best_fitness_history": neuro_agent.fitness_history,
                    "current_gen_best": max(neuro_agent.fitness_scores) if neuro_agent.fitness_scores else 0
                }
            }
            
            ppo_agent.save_model()
            q_agent.save_model()
            
            with open("report.json", "w") as f:
                json.dump(report, f, indent=4)
                
            running = False
        
        # Determine leader
        max_score = -1
        leader_idx = -1
        for i, game in enumerate(games):
            if game.score > max_score:
                max_score = game.score
                leader_idx = i
            elif game.score == max_score and max_score > 0:
                leader_idx = -1 # Tie or no score

        for i, game in enumerate(games):
            # Update dynamic stats
            game.stats["Time"] = time_str
            game.stats["Leader"] = (i == leader_idx)
            
            # Placeholder for model specific stats
            if i == 0: # PPO
                game.stats["Mode"] = "PPO"
            elif i == 1: # Q-Learning
                game.stats["LR"] = f"{q_agent.lr:.4f}"
                game.stats["Epsilon"] = f"{q_agent.epsilon:.2f}"
            elif i == 2: # Neuroevolution
                game.stats["Gen"] = str(neuro_agent.generation)
                game.stats["Ind"] = f"{neuro_agent.current_individual_idx + 1}/{neuro_agent.population_size}"
            else:
                game.stats["LR"] = "0.001" # Placeholder
                game.stats["Epsilon"] = "0.1" # Placeholder

            if not game.game_over:
                action = 0
                state_old = None
                prob = 0 # For PPO
                
                if i == 0:
                    # PPO Agent
                    state_old = game.get_state_vision()
                    action, prob = ppo_agent.get_action(state_old)
                    
                elif i == 1:
                    # Q-Learning Agent
                    state_old = game.get_state()
                    action = q_agent.get_action(state_old)
                    
                elif i == 2:
                    # Neuroevolution Agent
                    state_old = game.get_state()
                    action = neuro_agent.get_action(state_old)
                    
                else:
                    # Random action for others
                    action = random.randint(0, 3)

                # Execute Step if time is right
                current_time = pygame.time.get_ticks()
                if current_time - game.last_move_time >= game.move_interval:
                    game.last_move_time = current_time
                    
                    # For Q-Learning, we need the step result to train
                    reward, done, score = game.step(action)
                    
                    if i == 0: # Train PPO
                        ppo_agent.store_transition(state_old, action, prob, reward, done)
                        ppo_agent.train()
                    
                    if i == 1: # Train Q-Learning
                        state_new = game.get_state()
                        q_agent.train(state_old, action, reward, state_new, done)
            else:
                # Handle Game Over
                if i == 2: # Neuroevolution
                    # Record fitness (score + duration bonus maybe? for now just score)
                    # Let's add a small bonus for surviving to differentiate 0 scores
                    fitness = game.score * 100 + (game.snake_length) 
                    neuro_agent.update_fitness(fitness)
                    neuro_agent.next_individual()
                    game.reset()
                else:
                    game.reset()

            game.render(display)

        pygame.display.update()
        clock.tick(90)

    pygame.quit()
    quit()

if __name__ == "__main__":
    main()
