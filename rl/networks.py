import torch
import torch.nn as nn
import torch.nn.functional as F
class DuelingHead(nn.Module):
    def __init__(self, hidden: int, n_actions: int):
        super().__init__()
        self.V = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.A = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))
    def forward(self, h):
        V = self.V(h)
        A = self.A(h)
        return V + (A - A.mean(dim=1, keepdim=True))
class StrategicDQN(nn.Module):
    def __init__(self, in_dim: int, n_actions: int, dueling: bool=True):
        super().__init__()
        hid = 256
        self.body = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU()
        )
        self.head = DuelingHead(hid, n_actions) if dueling else nn.Linear(hid, n_actions)
    def forward(self, x):
        h = self.body(x)
        return self.head(h)

class TacticalDQN(nn.Module):
    def __init__(self, in_dim: int, n_actions: int, dueling: bool=True):
        super().__init__()
        hid = 256
        self.body = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU()
        )
        self.head = DuelingHead(hid, n_actions) if dueling else nn.Linear(hid, n_actions)
    def forward(self, x):
        h = self.body(x)
        return self.head(h)
