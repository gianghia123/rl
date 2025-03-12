from typing import Union
from enum import Enum, auto
from sys import stdout
from time import sleep
from itertools import cycle
from threading import Thread
from shutil import get_terminal_size

import numpy as np
import matplotlib.pyplot as plt
import random

from maze import Status, Enviroment

class Animate:
    def __init__(self, desc: str, end: str, sleep_time: float) -> None:
        self.desc = desc
        self.end = end
        self.sleep_time = sleep_time

        self.thread = Thread(target = self._animate, daemon = True)
        self.step = ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"]
        self.done = False

    def __enter__(self) -> None:
        self.thread.start()

    def __exit__(self, a, b, c) -> None:
        self.done = True
        cols = get_terminal_size((80, 20)).columns
        print('\r' + ' ' * cols, end = '', flush = True)
        print(f'\r{self.end}', flush = True)

    def _animate(self) -> None:
        for char in cycle(self.step):
            if self.done:
                break
            print(f'\r{self.desc} {char}', flush = True, end = '')
            sleep(self.sleep_time)

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
    
    def mc_approx_values(self) -> None:
        returns = [[] for i in range(25)]
        with Animate("[+] Training via MC.", "[!] End training", 0.1):
            for i in range(50000):
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
                            neighbors[j] = -1e9
                    self.policy[i] = np.exp(neighbors) / np.sum(np.exp(neighbors))
    
    def dp_approx_values(self, gamma) -> None:
        delta = -1e9
        with Animate("[+] Training via DP", "[!] End training", 0.1):
            while delta <= 1e-5:
                # Policy eval
                for i in range(24):
                    x, y = i % 5, i // 5

                    temp = self.values[i]
                    temp2 = 0
                    for dx, dy, dirc in [(0, -1, 0), (0, 1, 1), (-1, 0, 2), (1, 0, 3)]:
                        nx, ny = x + dx, y + dy
                        if (nx < 0 or nx >= 5 or ny < 0 or ny >= 5):
                            continue
                        else:
                            nindx = ny * 5 + nx
                            reward = -1 if nindx != 24 else 0
                            temp2 += self.policy[nindx][dirc] * (reward + gamma * self.values[nindx])
                    self.values[i] = temp2
                    delta = max(delta, abs(temp - temp2))

                # Policy update
                for i in range(24):
                    neighbors = np.array([0, 0, 0, 0])
                    x, y = i % 5, i // 5
                    for dx, dy, dirc in [(0, -1, 0), (0, 1, 1), (-1, 0, 2), (1, 0, 3)]:
                        nx, ny = x + dx, y + dy
                        if (nx < 0 or nx >= 5 or ny < 0 or ny >= 5):
                            neighbors[dirc] = -1e9
                        else:
                            neighbors[dirc] = self.values[ny * 5 + nx]
                    self.policy[i] = np.exp(neighbors) / np.sum(np.exp(neighbors))


if __name__ == "__main__":
    agent_a = Agent(b'imbored')
    print(agent_a.envi.envi)
    agent_a.dp_approx_values(0.015)
    agent_a.vis_pol()
    
