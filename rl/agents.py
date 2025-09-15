import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
from ..config import RL
Transition = namedtuple("Transition", "s a r s2 d")
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)
    def push(self, s, a, r, s2, d):
        self.buf.append(Transition(s,a,r,s2,d))
    def sample(self, batch_size: int):
        idx = np.random.choice(len(self.buf), batch_size, replace=False)
        batch = [self.buf[i] for i in idx]
        s  = np.stack([b.s  for b in batch])
        a  = np.stack([b.a  for b in batch])
        r  = np.stack([b.r  for b in batch])
        s2 = np.stack([b.s2 for b in batch])
        d  = np.stack([b.d  for b in batch]).astype(np.float32)
        return s, a, r, s2, d
    def __len__(self):
        return len(self.buf)

class NStepHelper:
    def __init__(self, n: int, gamma: float):
        self.n = max(1, n)
        self.gamma = gamma
        self.q = deque()
    def push(self, s, a, r):
        self.q.append((s,a,r))
    def pop_ready(self, s_next, done):
        out = []
        while len(self.q) >= self.n or (done and len(self.q) > 0):
            R = 0.0
            g = 1.0
            for i in range(min(self.n, len(self.q))):
                R += g * self.q[i][2]
                g *= self.gamma
            s0, a0, _ = self.q[0]
            out.append((s0, a0, R, s_next, done))
            self.q.popleft()
            if not done and len(self.q) < self.n:
                break
        return out
    def reset(self):
        self.q.clear()

class DQNAgent:
    def __init__(self, qnet: nn.Module, obs_dim: int, n_actions: int, cfg=RL, device=None):
        self.q = qnet
        self.target = type(qnet)(obs_dim, n_actions, getattr(cfg, "dueling", True))
        self.target.load_state_dict(self.q.state_dict())
        self.target.eval()
        self.n_actions = n_actions
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.q.to(self.device); self.target.to(self.device)
        self.opt = optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_size)
        self.gamma = cfg.gamma
        self.grad_clip = cfg.grad_clip
        self.target_update_every = cfg.target_update_every
        self.double = cfg.double_dqn
        self.n_step = NStepHelper(cfg.n_step, cfg.gamma)
        self.eps = cfg.eps_start
        self.eps_end = cfg.eps_end
        self.eps_decay_steps = cfg.eps_decay_steps
        self.step_i = 0

    def act(self, obs: np.ndarray, exploit: bool=False) -> int:
        self.step_i += 1
        self.eps = max(self.eps_end, self.eps - (1.0 - self.eps_end)/self.eps_decay_steps)
        if (not exploit) and (np.random.rand() < self.eps):
            return np.random.randint(self.n_actions)
        with torch.no_grad():
            q = self.q(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
            return int(q.argmax(dim=-1).item())

    def push(self, s, a, r, s2, d):
        self.n_step.push(s, a, r)
        ready = self.n_step.pop_ready(s2, d)
        for (s0, a0, Rn, sN, dN) in ready:
            self.buffer.push(s0, a0, Rn, sN, dN)
        if d:
            self.n_step.reset()

    def update(self, batch_size: int):
        if len(self.buffer) < batch_size:
            return 0.0
        s, a, r, s2, d = self.buffer.sample(batch_size)
        s  = torch.as_tensor(s,  dtype=torch.float32, device=self.device)
        a  = torch.as_tensor(a,  dtype=torch.int64,   device=self.device).unsqueeze(-1)
        r  = torch.as_tensor(r,  dtype=torch.float32, device=self.device).unsqueeze(-1)
        s2 = torch.as_tensor(s2, dtype=torch.float32, device=self.device)
        d  = torch.as_tensor(d,  dtype=torch.float32, device=self.device).unsqueeze(-1)
        with torch.no_grad():
            if self.double:
                a2 = self.q(s2).argmax(dim=-1, keepdim=True)
                q2 = self.target(s2).gather(1, a2)
            else:
                q2 = self.target(s2).max(dim=-1, keepdim=True)[0]
            y = r + (1.0 - d) * (self.gamma ** max(1, self.n_step.n)) * q2
        q = self.q(s).gather(dim=1, index=a)
        loss = (q - y).pow(2).mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)
        self.opt.step()
        if self.step_i % self.target_update_every == 0:
            self.target.load_state_dict(self.q.state_dict())
        return float(loss.item())
