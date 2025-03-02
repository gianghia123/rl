from typing import Union
from enum import Enum, auto

import numpy as np
import matplotlib.pyplot as plt
import random

from tqdm import trange

from maze import Status, Enviroment

class Agent:
    def __init__(self, seed: Union[str, bytes, int]) -> None:
        self.values = np.full((5 * 5), 0, dtype = np.float64)
        self.policy = np.full((5 * 5, 4), 1, dtype = np.float64)
        # self.policy[99, ::] = [0, 0, 0, 0]
        self.envi = Enviroment(seed = seed)

    def vis_pol(self) -> None:
        self.envi.reset()
        status = Status.IN_PROGRESS
        current_loc = 0
        path = [0]
        while status is Status.IN_PROGRESS:
            d = random.choices(['n', 's', 'w', 'e'], self.policy[current_loc])[0]
            status, new_loc = self.envi.step(d)
            path.append(new_loc)
        print(path)
        print(self.envi.trace_path(path))

    def trace(self) -> dict:
        self.envi.reset()
        state_reward = {}
        visited = set()
        current_loc = 0
        status = Status.IN_PROGRESS
        while status is Status.IN_PROGRESS:
            # Determine, from the current policy, a move for the current state
            move = random.choices(['n', 's', 'w', 'e'], weights = self.policy[current_loc], k = 1)[0]
            status, reward, new_loc = self.envi.next(move)
            if current_loc not in visited:
                visited.add(current_loc)
                state_reward[current_loc] = reward
            current_loc = new_loc
        return state_reward
    
    def approx_values(self) -> None:
        returns = [[] for i in range(25)]
        for i in trange(50000):
            rewards = self.trace()
            for s in rewards:
                returns[s].append(rewards[s])
                self.values[s] = np.mean(returns[s])
            temp = self.values.copy()
            for i in range(25):
                neighbors = np.zeros(4)
                x, y = i % 5, i // 5
                for j, (dx, dy) in enumerate([(0, -1), (0, 1), (-1, 0), (1, 0)]):
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < 5 and 0 <= ny < 5):
                        neighbors[j] = (self.values[ny * 5 + nx])
                    else:
                        pass
                self.policy[i] = np.exp(neighbors) / np.sum(np.exp(neighbors))

if __name__ == "__main__":
    agent = Agent(b'imbored')
    print(agent.envi.envi)
    agent.approx_values()
    agent.vis_pol()
    print(agent.policy)
    print(agent.values)
