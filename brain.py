import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MineSweeperCNN(nn.Module):
    """Actor-critic CNN (unchanged architecture from the original project)."""

    def __init__(self, grid_size=4):
        super().__init__()
        self.grid_size = grid_size
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.actor_fc = nn.Linear(64 * grid_size * grid_size, grid_size * grid_size)
        self.critic_fc = nn.Linear(64 * grid_size * grid_size, 1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        elif x.dim() == 3:
            x = x.unsqueeze(1)               # [B, 1, H, W]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        return self.actor_fc(x), self.critic_fc(x)


class FlyConnectomeNet(nn.Module):
    """
    16 sensory inputs -> 128 hidden -> 16 action logits (+ 1 value).

    fc1 (inputs -> hidden) and actor_fc (hidden -> actions) can be initialised
    from a FlyWire synapse-count matrix; see load_connectome().

    Inputs are re-encoded as (obs + 2) / 10 so that hidden cells are 0 and
    revealed numbers are positive. Without this, the all-positive connectome
    weights multiplied by the -2 "hidden" value would push almost every
    hidden unit below zero and the ReLUs would start out dead.
    """

    def __init__(self, grid_size=4, hidden=128):
        super().__init__()
        n = grid_size * grid_size
        self.grid_size = grid_size
        self.fc1 = nn.Linear(n, hidden)
        self.actor_fc = nn.Linear(hidden, n)
        self.critic_fc = nn.Linear(hidden, 1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = x.reshape(x.size(0), -1)
        x = (x + 2.0) / 10.0
        h = F.relu(self.fc1(x))
        return self.actor_fc(h), self.critic_fc(h)

    @staticmethod
    def _match_default_scale(w):
        """Rescale so the std matches PyTorch's default Linear init for this fan-in.
        Keeps the comparison with random init fair (same magnitude, different structure)."""
        fan_in = w.shape[1]
        target_std = 1.0 / math.sqrt(3.0 * fan_in)
        std = w.std()
        return w * (target_std / std) if std > 0 else w

    def load_connectome(self, path, mode="connectome", freeze=False, seed=0):
        """
        mode = "connectome": use the real wiring.
        mode = "shuffled":   same weight values, randomly permuted positions.
                             This is the control that tests whether the *wiring*
                             matters, not just the weight distribution.
        """
        blob = torch.load(path, weights_only=True)
        w1 = blob["W1"].float()  # [128, 16]
        w2 = blob["W2"].float()  # [16, 128]

        if mode == "shuffled":
            g = torch.Generator().manual_seed(seed)
            w1 = w1.flatten()[torch.randperm(w1.numel(), generator=g)].reshape(w1.shape)
            w2 = w2.flatten()[torch.randperm(w2.numel(), generator=g)].reshape(w2.shape)

        with torch.no_grad():
            self.fc1.weight.copy_(self._match_default_scale(w1))
            self.actor_fc.weight.copy_(self._match_default_scale(w2))

        if freeze:
            self.fc1.weight.requires_grad_(False)
            self.actor_fc.weight.requires_grad_(False)


def build_model(arch="cnn", grid_size=4, init="random",
                connectome_path="fly_connectome.pt", freeze=False, seed=0):
    """
    arch: "cnn" or "fly"
    init (fly only): "random", "connectome", or "shuffled"
    """
    if arch == "cnn":
        return MineSweeperCNN(grid_size=grid_size)

    if arch == "fly":
        if grid_size != 4:
            raise ValueError("The fly network has 16 sensory inputs, so it only supports a 4x4 grid.")
        model = FlyConnectomeNet(grid_size=grid_size)
        if init in ("connectome", "shuffled"):
            model.load_connectome(connectome_path, mode=init, freeze=freeze, seed=seed)
        return model

    raise ValueError(f"Unknown arch '{arch}'")