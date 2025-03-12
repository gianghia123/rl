from typing import Union

import numpy as np
import random
from enum import Enum, auto

from tqdm import trange

# random.seed(b"A love for my girlfriend")

def union(a, b):
    result = ['', '', '']
    for l in range(3):
        for j in range(3):
            result[l] += '#' if a[l][j] == '#' or b[l][j] == '#' else ' '
    return result

class Maze:
    def __init__(self, maz_size: int, seed: bytes = None) -> None:
        self.maz_size = maz_size
        self.maze = np.ones((self.maz_size ** 2, 4))
        # Directions: 0 -> North
        #             1 -> South
        #             2 -> West
        #             3 -> East

        random.seed(seed)

        # Maze generation
        stack = [0]
        visited = set()
        visited.add(0)
        while len(stack) != 0:
            random_action = random.choices(['n', 'o', 'r'], [30, 10, 60])
            if random_action == 'n':
                v = stack.pop()
            elif random_action == 'o':
                v = stack.pop(0)
            else:
                indx = random.randint(0, len(stack) - 1)
                v = stack.pop(indx)

            x = v % self.maz_size
            y = v // self.maz_size
            possible_neighbors = []
            for dx, dy, d in [(-1, 0, 'w'), (1, 0, 'e'), (0, 1, 's'), (0, -1, 'n')]:
                nx = x + dx
                ny = y + dy
                if (nx < 0 or nx >= self.maz_size or
                    ny < 0 or ny >= self.maz_size or
                    ny * self.maz_size + nx in visited):
                    continue
                possible_neighbors.append((ny * self.maz_size + nx, d))
            if len(possible_neighbors) == 0:
                continue
            else:
                stack.append(v)
                chosen_one, d = random.choice(possible_neighbors)
                stack.append(chosen_one)
                visited.add(chosen_one)
                match d:
                    case 'w':
                        self.maze[v,          2] = 0
                        self.maze[chosen_one, 3] = 0
                    case 'e':
                        self.maze[v,          3] = 0
                        self.maze[chosen_one, 2] = 0
                    case 's':
                        self.maze[v,          1] = 0
                        self.maze[chosen_one, 0] = 0
                    case 'n':
                        self.maze[v,          0] = 0
                        self.maze[chosen_one, 1] = 0

    def __str__(self) -> str:
        building_block = [
            ["###", "   ", "   "],
            ["   ", "   ", "###"],
            ["#  ", "#  ", "#  "],
            ["  #", '  #', '  #']
        ]
        result = []
        for y in range(self.maz_size):
            temp = ['', '', '']
            for x in range(self.maz_size):
                i = y * self.maz_size + x
                wall = self.maze[i]
                cell = ['   ', '   ', '   ']
                if wall[0] == 1:
                    cell = union(cell, building_block[0])
                if wall[1] == 1:
                    cell = union(cell, building_block[1])
                if wall[2] == 1:
                    cell = union(cell, building_block[2])
                if wall[3] == 1:
                    cell = union(cell, building_block[3])
                
                if i == 0:
                    cell[1] = cell[1][0] + 'S' + cell[1][2]
                if i == (self.maz_size ** 2) - 1:
                    cell[1] = cell[1][0] + 'E' + cell[1][2]

                temp[0] += cell[0]
                temp[1] += cell[1]
                temp[2] += cell[2]
            result.extend(temp)
        return '\n'.join(result)

class Status(Enum):
    IN_PROGRESS = 1
    HALTED = 2
    TERMINATED = 3

class Enviroment:
    def __init__(self, max_step: int = 2000, seed: Union[str, bytes, int] = None) -> None:
        self.envi = Maze(5, seed)
        self.current_loc = 0
        self.x = self.current_loc % self.envi.maz_size
        self.y = self.current_loc // self.envi.maz_size
        self.max_step = max_step
        self.current_step = 0
        self.current_status = Status.IN_PROGRESS

    def update_loc(self) -> None:
        self.x = self.current_loc % self.envi.maz_size
        self.y = self.current_loc // self.envi.maz_size

    def next(self, action: str) -> tuple[Status, int, int]:
        if self.current_status == Status.TERMINATED:
            raise ValueError("Enviroment has terminated.")
        elif self.current_status == Status.HALTED:
            raise ValueError("Enviroment has halted")
        
        wall = self.envi.maze[self.current_loc]
        # status = self.current_status
        match action:
            case 'n':
                dx, dy, wd = (0, -1, 0)
            case 's':
                dx, dy, wd = (0, 1, 1)
            case 'w':
                dx, dy, wd = (-1, 0, 2)
            case 'e':
                dx, dy, wd = (1, 0, 3)
        nx = self.x + dx
        ny = self.y + dy
        if (0 <= nx < self.envi.maz_size and 0 <= ny < self.envi.maz_size
            and wall[wd] == 0 and self.current_step < self.max_step):
            # Can update the location (the move is valid)
            self.current_loc = ny * self.envi.maz_size + nx
            self.update_loc()
            if self.current_loc == self.envi.maz_size ** 2 - 1:
                reward = 0
                self.current_status = Status.TERMINATED
            else:
                reward = -1
        elif self.current_step >= self.max_step:
            # End of live
            reward = -1
            self.current_status = Status.HALTED
        else:
            # Stuck!
            reward = -5
        self.current_step += 1
        return (self.current_status, reward, self.current_loc)

    def step(self, action: str) -> tuple[Status, int]:
        if self.current_status == Status.TERMINATED:
            raise ValueError("Enviroment has terminated!") 
        elif self.current_status == Status.HALTED:
            return (Status.TERMINATED, self.current_loc)
        
        wall = self.envi.maze[self.current_loc]
        match action:
            case 'n':
                dx, dy, wd = (0, -1, 0)
            case 's':
                dx, dy, wd = (0, 1, 1)
            case 'w':
                dx, dy, wd = (-1, 0, 2)
            case 'e':
                dx, dy, wd = (1, 0, 3)
        nx = self.x + dx
        ny = self.y + dy
        if (0 <= nx < self.envi.maz_size and 0 <= ny < self.envi.maz_size
            and wall[wd] == 0 and self.current_step < self.max_step):
            # Can update the location (the move is valid)
            self.current_loc = ny * self.envi.maz_size + nx
            self.update_loc()
            if self.current_loc == self.envi.maz_size ** 2 - 1:
                self.current_status = Status.TERMINATED
        elif self.current_step >= self.max_step:
            # End of live
            self.current_status = Status.HALTED
        else:
            # Stuck!
            pass
        self.current_step += 1
        return (self.current_status, self.current_loc)
    
    def reset(self) -> None:
        self.current_loc = 0
        self.update_loc()
        self.current_step = 0
        self.current_status = Status.IN_PROGRESS
        return 0 

    def trace_path(self, path) -> str:
        # print(path)
        building_block = [
            ["###", "   ", "   "],
            ["   ", "   ", "###"],
            ["#  ", "#  ", "#  "],
            ["  #", '  #', '  #']
        ]
        result = []
        for y in range(self.envi.maz_size):
            temp = ['', '', '']
            for x in range(self.envi.maz_size):
                i = y * self.envi.maz_size + x
                wall = self.envi.maze[i]
                cell = ['   ', '   ', '   ']
                if wall[0] == 1:
                    cell = union(cell, building_block[0])
                if wall[1] == 1:
                    cell = union(cell, building_block[1])
                if wall[2] == 1:
                    cell = union(cell, building_block[2])
                if wall[3] == 1:
                    cell = union(cell, building_block[3])
                
                if i == 0:
                    cell[1] = cell[1][0] + 'S' + cell[1][2]
                elif i == (self.envi.maz_size ** 2) - 1:
                    cell[1] = cell[1][0] + 'E' + cell[1][2]
                elif i in path:
                    cell[1] = cell[1][0] + '*' + cell[1][2]
                
                temp[0] += cell[0]
                temp[1] += cell[1]
                temp[2] += cell[2]
            result.extend(temp)
        return '\n'.join(result)

def test():
    a = Enviroment(seed = b'imbored')
    path = []
    for i in range(2000):
        try:
            move = random.choice(['n', 's', 'e', 'w'])
            status, _, cur_loc = a.next(move) 
            path.append(cur_loc)
            print(status, cur_loc)
            if status == Status.TERMINATED:
                break
        except ValueError:
            break
    print(path)
    print(a.trace_path(path))

if __name__ == "__main__":
    test()
