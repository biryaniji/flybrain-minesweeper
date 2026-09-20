import torch
import torch.nn as nn
import torch.nn.functional as F

class MineSweeperCNN(nn.Module):
    def __init__(self, grid_size=4):
        super(MineSweeperCNN, self).__init__()
        self.grid_size = grid_size
        
        # Spatial feature extraction via Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Actor head: Outputs logits for every cell on the grid
        self.actor_fc = nn.Linear(64 * grid_size * grid_size, grid_size * grid_size)
        
        # Critic head: Outputs the state value estimate
        self.critic_fc = nn.Linear(64 * grid_size * grid_size, 1)

    def forward(self, x):
        # Handle single state tensor or batched tensors
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]
        elif x.dim() == 3:
            x = x.unsqueeze(1)               # Shape: [Batch, 1, H, W]
            
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        x = x.reshape(x.size(0), -1)
        
        action_logits = self.actor_fc(x)
        state_value = self.critic_fc(x)
        
        return action_logits, state_value