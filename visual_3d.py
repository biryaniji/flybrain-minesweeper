import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import math
from environment import MinesweeperEnv
from ursina import *

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(SCRIPT_DIR, 'minesweeper_cnn.pt')

class MineSweeperCNN(nn.Module):
    def __init__(self, grid_size=4):
        super(MineSweeperCNN, self).__init__()
        self.grid_size = grid_size
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.actor_fc = nn.Linear(64 * grid_size * grid_size, grid_size * grid_size)
        self.critic_fc = nn.Linear(64 * grid_size * grid_size, 1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        action_logits = self.actor_fc(x)
        state_value = self.critic_fc(x)
        return action_logits, state_value

app = Ursina()
window.title = "Agentic Fly Playthrough"
window.borderless = False
window.color = color.rgb(15, 15, 18) # Deep studio gray/black

# Cinematic Lighting Setup
DirectionalLight(y=4, z=2, shadows=True, color=color.rgb(200, 200, 220))
PointLight(y=5, x=0, z=0, color=color.white) # Overhead spot
AmbientLight(color=color.rgba(80, 80, 90, 255))

# 85mm Lens Aesthetic: Narrow FOV, pulled back to compress the scene
camera.orthographic = False
camera.fov = 35 
camera.position = (1.5, 9, -11)
camera.rotation = (45, 0, 0)

# Constructing the AI Fly Agent
class AIFly(Entity):
    def __init__(self):
        super().__init__(position=(0, 1.5, 0))
        
        # Body
        self.body = Entity(parent=self, model='sphere', color=color.black, scale=(0.3, 0.2, 0.4))
        self.eye_l = Entity(parent=self.body, model='sphere', color=color.red, scale=0.4, position=(-0.3, 0.2, 0.3))
        self.eye_r = Entity(parent=self.body, model='sphere', color=color.red, scale=0.4, position=(0.3, 0.2, 0.3))
        
        # Wings
        self.wing_l = Entity(parent=self.body, model='cube', color=color.white50, scale=(0.8, 0.05, 0.4), position=(-0.4, 0.3, -0.2))
        self.wing_r = Entity(parent=self.body, model='cube', color=color.white50, scale=(0.8, 0.05, 0.4), position=(0.4, 0.3, -0.2))

    def update(self):
        # High-speed wing flap physics using a sine wave based on system time
        flap_angle = math.sin(time.time() * 60) * 30
        self.wing_l.rotation_z = flap_angle
        self.wing_r.rotation_z = -flap_angle

class Game3DController(Entity):
    def __init__(self):
        super().__init__()
        self.grid_size = 4
        self.env = MinesweeperEnv(size=self.grid_size, n_mines=2)

        self.ai_brain = MineSweeperCNN(grid_size=self.grid_size)
        assert self.ai_brain is not None, "FATAL: self.ai_brain is None! Save your file!"

        if not os.path.exists(WEIGHTS_PATH):
            print(f"Error: '{WEIGHTS_PATH}' not found. Run training first!")
            application.quit()
            sys.exit(1)

        self.ai_brain.load_state_dict(torch.load(WEIGHTS_PATH, weights_only=True))
        self.ai_brain.eval()
        
        # Studio Ground Plane
        Entity(model='plane', scale=20, color=color.rgb(25, 25, 30), position=(1.5, -0.5, -1.5))
        
        self.fly = AIFly()
        self.tile_entities = []
        self.create_board_3d()

        self.state = self.env.reset()
        self.done = False
        self.timer = 0
        self.game_reset_delay = 0

    def create_board_3d(self):
        for row in self.tile_entities:
            for tile in row:
                destroy(tile)

        self.tile_entities = []
        offset = (self.grid_size - 1) / 2.0

        for r in range(self.grid_size):
            row_tiles = []
            for c in range(self.grid_size):
                tile = Entity(
                    model='cube',
                    color=color.rgb(200, 200, 200), # Clean matte white hidden tiles
                    scale=(0.85, 0.2, 0.85),
                    position=(c - offset, 0, offset - r),
                    collider='box'
                )
                row_tiles.append(tile)
            self.tile_entities.append(row_tiles)

        # Reset fly position on new board
        self.fly.position = (0, 3, 0)

    def update(self):
        if self.done:
            self.game_reset_delay += time.dt
            if self.game_reset_delay > 2.5: # Slightly longer pause to view the win state
                self.state = self.env.reset()
                self.create_board_3d()
                self.done = False
                self.game_reset_delay = 0
            return

        self.timer += time.dt
        if self.timer > 1.2: # Slowed down slightly so you can watch the fly travel
            self.timer = 0

            state_tensor = torch.FloatTensor(self.state)
            with torch.no_grad():
                action_logits, _ = self.ai_brain(state_tensor)

            valid_actions = self.env.get_valid_actions()
            action_logits = action_logits.squeeze(0)
            action_logits[~torch.tensor(valid_actions, dtype=torch.bool)] = -1e9

            action = torch.argmax(action_logits).item()
            r, c = divmod(action, self.grid_size)
            
            # Animate the fly darting to the chosen tile
            offset = (self.grid_size - 1) / 2.0
            target_x = c - offset
            target_z = offset - r
            self.fly.animate_position((target_x, 0.6, target_z), duration=0.3, curve=curve.in_out_quad)

            self.state, reward, self.done, won = self.env.step(action)
            obs = self.env.get_observation()
            
            for row_idx in range(self.grid_size):
                for col_idx in range(self.grid_size):
                    val = obs[row_idx, col_idx]
                    tile = self.tile_entities[row_idx][col_idx]

                    if val == -2:
                        pass # Keep matte white
                    elif val == 0:
                        tile.color = color.rgb(40, 40, 45) # Sleek dark gray
                        tile.scale_y = 0.05
                    elif val > 0:
                        # Vibrant UI accents for numbers
                        tile.color = color.cyan if val == 1 else color.magenta 
                        tile.scale_y = 0.05

            if self.done:
                if won:
                    print("Fly AI Victory!")
                    self.fly.animate_position((1.5, 4, 1.5), duration=1.0) 
                    for row in self.tile_entities:
                        for tile in row:
                            tile.color = color.green # Fixed from springgreen
                else:
                    print("Fly AI hit a mine!")
                    self.fly.animate_rotation((180, 0, 0), duration=0.5) 
                    for row in self.tile_entities:
                        for tile in row:
                            tile.color = color.red # Fixed from crimson
controller = Game3DController()
app.run()