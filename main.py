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

from models.hamiltonian import HamiltonianAgent
from models.qlearning import QLearningAgent

# Initialize 3 games
games = [
    SnakeGame(GAME_WIDTH, HEIGHT, 0, "Hamiltonian Cycle"),
    SnakeGame(GAME_WIDTH, HEIGHT, 300, "Q-Learning"),
    SnakeGame(GAME_WIDTH, HEIGHT, 600, "Model 3")
]

# Initialize Agents
# Grid size is 300x600, block size 10 -> 30x60 grid
hamiltonian_agent = HamiltonianAgent(30, 60)
q_agent = QLearningAgent(n_actions=4)

import time

def main():
    start_time = time.time()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        display.fill((0, 0, 0))
        
        # Calculate stats
        elapsed_time = int(time.time() - start_time)
        time_str = f"{elapsed_time // 60:02d}:{elapsed_time % 60:02d}"
        
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
            if i == 0: # Hamiltonian
                game.stats["Mode"] = "Deterministic"
            elif i == 1: # Q-Learning
                game.stats["LR"] = f"{q_agent.lr:.4f}"
                game.stats["Epsilon"] = f"{q_agent.epsilon:.2f}"
            else:
                game.stats["LR"] = "0.001" # Placeholder
                game.stats["Epsilon"] = "0.1" # Placeholder

            if not game.game_over:
                action = 0
                state_old = None
                
                if i == 0:
                    # Hamiltonian Agent
                    action = hamiltonian_agent.get_action((game.x, game.y), game.block_size)
                    
                elif i == 1:
                    # Q-Learning Agent
                    state_old = game.get_state()
                    action = q_agent.get_action(state_old)
                    
                    # We need to step and get reward
                    # But the loop structure was: check time -> step.
                    # We need to ensure we only train when a step actually happens.
                    # The previous logic had a time check inside the loop? 
                    # No, I removed it in the last refactor and put it in main?
                    # Wait, I need to check the time logic again.
                    
                    # Let's look at the previous main.py content via search or just assume standard loop.
                    # The previous main.py had:
                    # current_time = pygame.time.get_ticks()
                    # if current_time - game.last_move_time >= game.move_interval:
                    #    game.last_move_time = current_time
                    #    game.step(action)
                    
                    # I need to wrap the training logic in this time check.
                    pass # Logic handled below
                    
                else:
                    # Random action for others
                    action = random.randint(0, 3)
                    # game.step(action) # Logic handled below

                # Execute Step if time is right
                current_time = pygame.time.get_ticks()
                if current_time - game.last_move_time >= game.move_interval:
                    game.last_move_time = current_time
                    
                    # For Q-Learning, we need the step result to train
                    reward, done, score = game.step(action)
                    
                    if i == 1: # Train Q-Learning
                        state_new = game.get_state()
                        q_agent.train(state_old, action, reward, state_new, done)
            else:
                # Auto reset after a delay or just stay dead?
                # For testing random movement, let's auto reset immediately so we see action
                game.reset()

            game.render(display)

        pygame.display.update()
        clock.tick(90)

    pygame.quit()
    quit()

if __name__ == "__main__":
    main()
