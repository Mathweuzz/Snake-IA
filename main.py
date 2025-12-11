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

# Initialize 3 games
games = [
    SnakeGame(GAME_WIDTH, HEIGHT, 0, "Hamiltonian Cycle"),
    SnakeGame(GAME_WIDTH, HEIGHT, 300, "Model 2"),
    SnakeGame(GAME_WIDTH, HEIGHT, 600, "Model 3")
]

# Initialize Agents
# Grid size is 300x600, block size 10 -> 30x60 grid
hamiltonian_agent = HamiltonianAgent(30, 60)

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
            else:
                game.stats["LR"] = "0.001" # Placeholder
                game.stats["Epsilon"] = "0.1" # Placeholder

            if not game.game_over:
                action = 0
                if i == 0:
                    # Hamiltonian Agent
                    action = hamiltonian_agent.get_action((game.x, game.y), game.block_size)
                else:
                    # Random action for others
                    action = random.randint(0, 3)
                
                # We need to control the speed of updates, otherwise it's too fast
                # The game logic handles speed via internal checks if we wanted, 
                # but here we are calling step() every frame? 
                # Wait, the previous logic had time checks inside the loop.
                # The SnakeGame class I wrote doesn't have the time check in step().
                # I should add the time check logic back to the main loop or the game class.
                # Let's add it to the main loop for simplicity for now, or better, 
                # let's make the step() only execute if enough time passed.
                
                current_time = pygame.time.get_ticks()
                if current_time - game.last_move_time >= game.move_interval:
                    game.last_move_time = current_time
                    game.step(action)
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
