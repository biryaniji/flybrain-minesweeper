import numpy as np


class MinesweeperEnv:
    """
    Minesweeper environment.

    Observation: (size, size) float array.
        -2   = hidden cell
         0-8 = revealed cell showing its neighbouring mine count
        -1   = revealed mine (only visible after a loss)

    Fix vs. the original: mines are placed AFTER the first click, so the
    opening move can never be a mine (standard Minesweeper rule).
    """

    def __init__(self, size=4, n_mines=2, safe_first_click=True):
        self.size = size
        self.n_mines = n_mines
        self.safe_first_click = safe_first_click
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self):
        self.grid = np.zeros((self.size, self.size), dtype=int)
        self.revealed = np.zeros((self.size, self.size), dtype=bool)
        self.game_over = False
        self.won = False
        self.mines_placed = False

        if not self.safe_first_click:
            self._place_mines(exclude=None)

        return self.get_observation()

    def _place_mines(self, exclude=None):
        cells = np.arange(self.size * self.size)
        if exclude is not None:
            cells = cells[cells != exclude]

        indices = np.random.choice(cells, self.n_mines, replace=False)
        for idx in indices:
            r, c = divmod(int(idx), self.size)
            self.grid[r, c] = -1

        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r, c] == -1:
                    continue
                count = 0
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.size and 0 <= nc < self.size:
                            if self.grid[nr, nc] == -1:
                                count += 1
                self.grid[r, c] = count

        self.mines_placed = True

    # ------------------------------------------------------------ observation
    def get_observation(self):
        obs = np.full((self.size, self.size), -2, dtype=float)
        obs[self.revealed] = self.grid[self.revealed]
        return obs

    def get_valid_actions(self):
        # 1 = unrevealed (clickable), 0 = already revealed. Flat array of length size*size.
        return (~self.revealed).astype(int).flatten()

    # ------------------------------------------------------------------- step
    def step(self, action_idx):
        if self.game_over:
            return self.get_observation(), 0.0, True, self.won

        action_idx = int(action_idx)

        if not self.mines_placed:
            self._place_mines(exclude=action_idx)

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
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
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