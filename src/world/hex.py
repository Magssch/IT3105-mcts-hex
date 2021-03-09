from __future__ import annotations

from collections import deque
from typing import List, Optional, Set, Tuple

from src import parameters
from visualize import Visualize
from world.simulated_world import SimulatedWorld

# class Peg:

#     def __init__(self) -> None:
#         self.connect_pegs = set(self)

#     def connect(self, otherPeg: Peg):
#         self.connect_pegs.union(otherPeg.connect_pegs)
#         return self.connect_pegs


class Hex(SimulatedWorld):

    opposite_player = {
        1: 2,
        2: 1,
    }

    def __init__(self, state: Optional[Tuple[int, ...]]):
        self.__size: int = parameters.SIZE
        self.__length = self.__size ** 2

        self.__modified_list = {
            1: [False for _ in range(self.__size)],
            2: [False for _ in range(self.__size)]
        }
        self.__ending_indices = {
            1: set([self.__size * (i + 1) - 1 for i in range(self.__size)]),
            2: set([self.__length - (i + 1) for i in range(self.__size)]),
        }

        if state is None:
            self.__player_id, *self.__board = self.reset()
        else:
            self.__player_id, *self.__board = state

    def reset(self) -> Tuple[int, ...]:
        self.__player_id = 1
        self.__board = tuple(0 for _ in range(self.__length))
        return self.__get_state()

    def get_legal_actions(self) -> Tuple[int, ...]:
        actions = [i for i in range(self.__size)]
        return tuple(filter(lambda node: self.__board[node] > 0, actions))

    def generate_child_states(self) -> Tuple[Tuple[int, ...], ...]:
        child_states = []
        for i in self.get_legal_actions():
            child_states += self.generate_state(i)
        return tuple(child_states)

    def generate_state(self, action: int) -> Tuple[int, ...]:
        next_board = list(self.__board)
        next_board[action] = self.__player_id
        return (Hex.opposite_player[self.__player_id], *next_board)

    def is_final_state(self) -> bool:
        """
        Checks whether the current player has won the game.
        """
        if sum(self.__modified_list[self.__player_id]) < self.__size:  # Does the player have the sufficient amount of pegs along its axis?
            return False

        # Sufficient amount of pegs, check for path using BFS
        visited_cells = set()
        for i in range(self.__size):
            index = i if self.__player_id == 1 else i * self.__size
            if self.__board[index] == self.__player_id and self.__board[index] not in visited_cells:

                # BFS
                visited_cells.add(index)
                queue = deque()
                queue.append(index)
                while len(queue) > 0:
                    current_cell = queue.popleft()
                    for neighbour in self.__get_filled_neighbours(current_cell):
                        if neighbour not in visited_cells:
                            queue.append(neighbour)
                            if neighbour in self.__ending_indices[self.__player_id]:
                                return True
                    visited_cells.add(current_cell)
        return False

    def step(self, action: Tuple[int, int]) -> Tuple[int, ...]:
        index = self.__coordinates_to_index(action)
        assert 0 <= index < self.__size ** 2, 'Illegal action, index out of range'
        assert self.__board[index] == 0, 'Illegal action, cell is occupied'

        self.__board = tuple(self.__board[:index] + (self.__player_id,) + self.__board[index:])
        self.__modified_list[self.__player_id][self.__player_axis(action)] = True  # Used to speed up winning condition check
        self.__player_id = Hex.opposite_player[self.__player_id]
        return self.__get_state()

    def __get_state(self) -> Tuple[int, ...]:
        return (self.__player_id, *self.__board)

    def __player_axis(self, action: Tuple[int, int]) -> int:
        row, column = action
        return row * int(self.__player_id == 2) + column * int(self.__player_id == 1)

    def __coordinates_to_index(self, coordinates: Tuple[int, int]) -> int:
        return (coordinates[0] * self.__size) + coordinates[1]

    def __get_filled_neighbours(self, index: int) -> Set[int]:
        return set(filter(lambda cell: 0 <= cell < self.__length and self.__board[cell] == self.__player_id, self.__get_neighbouring_indices(index)))

    def __get_neighbouring_indices(self, index: int) -> Set[int]:
        return {
            index - self.__size,
            index + self.__size,
            index + 1,
            index - 1,
            index - self.__size + 1,
            index + self.__size - 1
        }
        """
        up = self.__board[index-self.__size]
        down = self.__board[index + self.__size]
        right = self.__board[index + 1]
        left = self.__board[index - 1]
        upright = self.__board[index-self.__size+1]
        # upleft = self.__board[index-self.__size-1]
        # downright = self.__board[index + self.__size+1]
        downleft = self.__board[index + self.__size-1]"""

    def __draw_board(self, action: int) -> None:
        Visualize.draw_board(self, self.__board, action)
