import pygame
import random
import numpy as np

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (213, 50, 80)
GREEN = (0, 255, 0)
BLUE = (50, 153, 213)
GRAY = (50, 50, 50)

class SnakeGame:
    def __init__(self, width, height, x_offset, title="Snake"):
        self.width = width
        self.height = height
        self.x_offset = x_offset
        self.title = title
        self.block_size = 10
        self.speed = 15
        
        self.font_style = pygame.font.SysFont("bahnschrift", 15)
        self.small_font = pygame.font.SysFont("bahnschrift", 12)
        
        self.best_score = 0
        self.deaths = 0
        self.stats = {} # Dictionary for dynamic stats (LR, Time, etc.)
        
        self.reset()

    def reset(self):
        if hasattr(self, 'score'):
            if self.score > self.best_score:
                self.best_score = self.score
        
        self.game_over = False
        self.x = self.width / 2
        self.y = self.height / 2
        self.x_change = 0
        self.y_change = 0
        self.snake_list = []
        self.snake_length = 1
        self.score = 0
        
        self.place_food()
        
        self.last_move_time = pygame.time.get_ticks()
        self.move_interval = 1000 / self.speed

    def place_food(self):
        self.foodx = round(random.randrange(0, self.width - self.block_size) / 10.0) * 10.0
        self.foody = round(random.randrange(0, self.height - self.block_size) / 10.0) * 10.0

    def step(self, action):
        # Action: 0=Left, 1=Right, 2=Up, 3=Down
        if action == 0:
            self.x_change = -self.block_size
            self.y_change = 0
        elif action == 1:
            self.x_change = self.block_size
            self.y_change = 0
        elif action == 2:
            self.y_change = -self.block_size
            self.x_change = 0
        elif action == 3:
            self.y_change = self.block_size
            self.x_change = 0
            
        # Move snake
        self.x += self.x_change
        self.y += self.y_change
        
        # Check boundaries
        if self.x >= self.width or self.x < 0 or self.y >= self.height or self.y < 0:
            self.game_over = True
            self.deaths += 1
            return -10, True, self.score
            
        # Update snake body
        snake_head = [self.x, self.y]
        self.snake_list.append(snake_head)
        if len(self.snake_list) > self.snake_length:
            del self.snake_list[0]
            
        # Check self collision
        for x in self.snake_list[:-1]:
            if x == snake_head:
                self.game_over = True
                self.deaths += 1
                return -10, True, self.score
                
        # Check food
        reward = 0
        if self.x == self.foodx and self.y == self.foody:
            self.foodx = round(random.randrange(0, self.width - self.block_size) / 10.0) * 10.0
            self.foody = round(random.randrange(0, self.height - self.block_size) / 10.0) * 10.0
            self.snake_length += 1
            self.score += 1
            reward = 10
            
        return reward, False, self.score

    def render(self, display):
        # Draw border
        border_color = GRAY
        if self.stats.get("Leader", False):
            border_color = (255, 215, 0) # Gold for leader
            
        pygame.draw.rect(display, border_color, [self.x_offset, 0, self.width, self.height], 2)
        
        # Draw Food
        pygame.draw.rect(display, RED, [self.x_offset + self.foodx, self.foody, self.block_size, self.block_size])
        
        # Draw Snake
        for x in self.snake_list:
            pygame.draw.rect(display, GREEN, [self.x_offset + x[0], x[1], self.block_size, self.block_size])
            
        # Draw Stats Overlay
        y_pos = 5
        line_height = 15
        
        # Title
        title_msg = self.font_style.render(f"{self.title}", True, WHITE)
        display.blit(title_msg, [self.x_offset + 5, y_pos])
        y_pos += 20
        
        # Score
        score_msg = self.small_font.render(f"Score: {self.score} (Best: {self.best_score})", True, WHITE)
        display.blit(score_msg, [self.x_offset + 5, y_pos])
        y_pos += line_height
        
        # Deaths/Epochs
        death_msg = self.small_font.render(f"Deaths: {self.deaths}", True, WHITE)
        display.blit(death_msg, [self.x_offset + 5, y_pos])
        y_pos += line_height
        
        # Custom Stats
        for key, value in self.stats.items():
            if key == "Leader": continue # Handled by border
            msg = self.small_font.render(f"{key}: {value}", True, WHITE)
            display.blit(msg, [self.x_offset + 5, y_pos])
            y_pos += line_height
        
        if self.game_over:
            msg = self.font_style.render("Game Over", True, RED)
            display.blit(msg, [self.x_offset + self.width/2 - 40, self.height/2])
