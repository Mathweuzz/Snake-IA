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

# Initialize 3 games
games = [
    SnakeGame(GAME_WIDTH, HEIGHT, 0, "Model 1"),
    SnakeGame(GAME_WIDTH, HEIGHT, 300, "Model 2"),
    SnakeGame(GAME_WIDTH, HEIGHT, 600, "Model 3")
]

def main():
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        display.fill((0, 0, 0))

        for i, game in enumerate(games):
            if not game.game_over:
                # Random action for now: 0=Left, 1=Right, 2=Up, 3=Down
                # To make it slightly less chaotic, only change direction occasionally
                # But for now, pure random is fine as requested
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
