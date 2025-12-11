import numpy as np

class HamiltonianAgent:
    def __init__(self, cols, rows):
        self.cols = cols
        self.rows = rows
        self.cycle = np.zeros((cols, rows), dtype=int)
        self.generate_cycle()

    def generate_cycle(self):
        # Action mapping: 0=Left, 1=Right, 2=Up, 3=Down
        
        # We assume cols is even.
        # Pattern:
        # Col 0: Down to rows-1
        # Col 1: Up to 1
        # Col 2: Down to rows-1
        # ...
        # Last Col: Up to 0
        # Row 0 (except (0,0)): Left to start
        
        for x in range(self.cols):
            for y in range(self.rows):
                # Default to None or error
                action = -1
                
                if x == 0:
                    if y == 0:
                        # Start point, actually we arrive here from (1,0)
                        # But if we are at (0,0), we go Down
                        action = 3 # Down
                    elif y < self.rows - 1:
                        action = 3 # Down
                    else:
                        action = 1 # Right to (1, rows-1)
                
                elif x % 2 == 0: # Even columns (2, 4, ...)
                    if y == 1:
                        action = 1 # Right
                    elif y < self.rows - 1:
                        action = 3 # Down
                    else:
                        action = 1 # Right
                        
                elif x % 2 != 0: # Odd columns (1, 3, ...)
                    if x == self.cols - 1: # Last column
                        if y > 0:
                            action = 2 # Up
                        else:
                            action = 0 # Left (at 0,0) -> back to (cols-2, 0)
                    else: # Normal odd column
                        if y > 1:
                            action = 2 # Up
                        elif y == 1:
                            action = 1 # Right
                        else:
                            # y=0, we should be moving Left here
                            action = 0 # Left
                            
                # Overwrite for the top row return path
                if y == 0 and x > 0:
                    action = 0 # Left
                    
                self.cycle[x, y] = action

    def get_action(self, head_pos, block_size):
        # Convert pixel coordinates to grid coordinates
        grid_x = int(head_pos[0] / block_size)
        grid_y = int(head_pos[1] / block_size)
        
        # Safety check
        if 0 <= grid_x < self.cols and 0 <= grid_y < self.rows:
            return self.cycle[grid_x, grid_y]
        else:
            return 0 # Should not happen if logic is correct
