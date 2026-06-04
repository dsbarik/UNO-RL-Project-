from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity=100000) -> None:
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done, mask, next_mask):
        self.buffer.append((state, action, reward, next_state, done, mask, next_mask))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        state, action, reward, next_state, done, mask, next_mask = zip(*batch)

        return (
            torch.FloatTensor(np.array(state)),
            torch.LongTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(np.array(next_state)),
            torch.FloatTensor(done),
            torch.FloatTensor(np.array(mask)),
            torch.FloatTensor(np.array(next_mask)),
        )
