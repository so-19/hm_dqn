import numpy as np
from collections import deque, namedtuple
import random
Transition = namedtuple("Transition", "s a r s2 d")
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, d):
        self.buf.append(Transition(s,a,r,s2,d))
    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s  = np.stack([b.s  for b in batch])
        a  = np.stack([b.a  for b in batch])
        r  = np.stack([b.r  for b in batch])
        s2 = np.stack([b.s2 for b in batch])
        d  = np.stack([b.d  for b in batch]).astype(np.float32)
        return s, a, r, s2, d
    def __len__(self):
        return len(self.buf)