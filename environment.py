import numpy as np

class MinesweeperEnv:
    def __init__(self, size=4, n_mines=2):
        self.size = size
        self.n_mines = n_mines
        self.reset()

    def reset(self):
        self.grid = np.zeros((self.size, self.size), dtype=int)
        self.revealed = np.zeros((self.size, self.size), dtype=bool)
        self.game_over = False
        self.won = False
        
        indices = np.random.choice(self.size * self.size, self.n_mines, replace=False)
        for idx in indices:
            r, c = divmod(idx, self.size)
            self.grid[r, c] = -1

        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r, c] == -1:
                    continue
                count = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.size and 0 <= nc < self.size:
                            if self.grid[nr, nc] == -1:
                                count += 1
                self.grid[r, c] = count
        return self.get_observation()

    def get_observation(self):
        obs = np.full((self.size, self.size), -2, dtype=float)
        obs[self.revealed] = self.grid[self.revealed]
        return obs

    def get_valid_actions(self):
        # Returns a 1D flat array of 1s (valid) for unrevealed cells and 0s (invalid) for revealed ones
        return (~self.revealed).astype(int).flatten()

    def step(self, action_idx):
        if self.game_over:
            return self.get_observation(), 0.0, True, self.won

        r, c = divmod(action_idx, self.size)
        
        if self.revealed[r, c]:
            return self.get_observation(), -0.2, False, False  

        self.revealed[r, c] = True

        if self.grid[r, c] == -1:
            self.game_over = True
            return self.get_observation(), -5.0, True, False

        if self.grid[r, c] == 0:
            queue = [(r, c)]
            while queue:
                cr, cc = queue.pop(0)
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < self.size and 0 <= nc < self.size and not self.revealed[nr, nc]:
                            self.revealed[nr, nc] = True
                            if self.grid[nr, nc] == 0:
                                queue.append((nr, nc))

        unrevealed_safe = np.sum(~self.revealed & (self.grid != -1))
        if unrevealed_safe == 0:
            self.game_over = True
            self.won = True
            return self.get_observation(), 10.0, True, True

        return self.get_observation(), 1.0, False, False