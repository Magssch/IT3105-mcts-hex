from typing import List, Set, Tuple

import numpy as np
from simulated_world import SimulatedWorld

from data_classes import Action, Shape
from visualize import Visualize
import ..parameters
from collections import deque

class Hex(SimulatedWorld):

    opposite_player = {
        1: 2,
        2: 1,
    }

    def __init__(self, state: Tuple[int]=None):
        self.__size: int = parameters.SIZE
        self.__length = se

        self.player_one_columns = [False for _ in range(self.__size)]
        self.player_two_rows = [False for _ in range(self.__size)]

        self.__ending_indices = (_, set([i for i in range(self.__size)]), set([i * self.__size for i in range(self.__size)]))

        if self.__board == None:
            self.__player_id, *self.__board = self.reset()
        else:
            self.__player_id, *self.__board = state
    
    def reset() -> Tuple[int]:
        self.__player_id = 1
        self.__board = tuple(0 for _ in self.__size ** 2)
        return self.__get_state()

    def generate_child_states(state: Tuple[int]) -> Tuple[Tuple[int]]:
        pass
    
    def is_final_state() -> bool:
        """
        Checks whether the current player has won the game.
        """
        # Only do BFS if player has sufficient number of pegs in rows for a possible win
        if self.__player_id == 1:
            if sum(self.__player_two_rows) < self.__size:
                return False
        else:
            if sum(self.__player_one_columns) < self.__size:
                return False

        visited_cells = set()
        for i in range(self.__size):
            index = i if self.__player_id == 1 else i * self.__size
            if self.__board[index] == self.__player_id and self.__board[index] not in visited_cells:
                
                # BFS
                visited_cells.add(index)
                queue = deque()
                queue.append(index)
                while len(queue) > 0:
                    current_node = queue.popleft()
                    for neighbour in self.__get_neighbouring_nodes(current_node):
                        if neighbour not in visited_cells:
                            queue.append(neighbour)
                            if neighbour in self.__ending_indices[self.__player_id]:
                                return True
                    visited_cells.add(current_node) 
        return False
                


        

    
    def step(action: Tuple[int, int]) -> Tuple[int]:
        index = self.__coordinates_to_index(action)
        assert 0 <= index < self.__size ** 2, 'Index out of range'
        assert self.__board[action] == 0
        
        self.__board[index] = self.__player_id
        self.__player_id = Hex.opposite_player[self.__player_id]
        return self.__get_state()
        
    def __get_state() -> Tuple[int]:
        return (self.__player_id, **self.__board)

    def __coordinates_to_index(coordinates: Tuple[int, int]) -> int:
        return (coordinates[0] * self.size) + coordinates[1]
        
    def __get_filled_neighbours(index: int) -> Set[int, ...]:
        return set(filter(lambda node: 0 <= node < self.__size ** 2 and self.board[node] == self.__player_id, self.__get_neighbouring_nodes))
    
    def __get_neighbouring_indices(index: int) -> Set[int, ...]:
            return {
                index-self.__size,
                index + self.__size, 
                index + 1,
                index - 1,
                index-self.__size+1,
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


    def __get_board(self):
        return self.__board

    def __draw_board(self, action: Action) -> None:
        Visualize.draw_board(self.__board_type, self.__board, action.positions)

    def __make_move(self, action: Action, visualize: bool) -> None:
        if self.__is_legal_action(action):
            self.__board[action.start_coordinates] = 2
            self.__board[action.adjacent_coordinates] = 2
            self.__board[action.landing_coordinates] = 1

            if visualize:
                self.__draw_board(action)

    def __pegs_remaining(self) -> int:
        return (self.__board == 1).sum()

    def __game_over(self) -> bool:
        return len(self.get_all_legal_actions()) < 1

    def __is_legal_action(self, action: Action) -> bool:
        return self.__action_is_inside_board(action) \
            and self.__cell_contains_peg(action.adjacent_coordinates) \
            and not self.__cell_contains_peg(action.landing_coordinates)

    def __cell_contains_peg(self, coordinates: Tuple[int, int]) -> bool:
        return self.__board[coordinates] == 1

    def __action_is_inside_board(self, action: Action) -> bool:
        return (action.adjacent_coordinates[0] >= 0 and action.adjacent_coordinates[0] < self.__size) \
            and (action.adjacent_coordinates[1] >= 0 and action.adjacent_coordinates[1] < self.__size) \
            and (action.landing_coordinates[0] >= 0 and action.landing_coordinates[0] < self.__size) \
            and (action.landing_coordinates[1] >= 0 and action.landing_coordinates[1] < self.__size) \
            and self.__board[action.adjacent_coordinates] != 0 and self.__board[action.landing_coordinates] != 0

    def __get_legal_actions_for_coordinates(self, coordinates: Tuple[int, int]) -> Tuple[Action]:
        legal_actions: List[Action] = []
        for direction_vector in self._edges:
            action = Action(coordinates, direction_vector)
            if self.__is_legal_action(action):
                legal_actions.append(action)
        return tuple(legal_actions)

    def __get_all_legal_actions(self) -> Tuple[Action]:
        legal_actions: List[Action] = []
        for i in range(self.__board.shape[0]):
            for j in range(self.__board.shape[0]):
                if self.__cell_contains_peg((i, j)):
                    legal_actions_for_position = self.__get_legal_actions_for_coordinates((i, j))
                    if len(legal_actions_for_position) > 0:
                        legal_actions += legal_actions_for_position
        return tuple(legal_actions)

    def __str__(self):
        return str(self.__board)
