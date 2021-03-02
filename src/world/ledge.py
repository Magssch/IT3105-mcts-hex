from typing import List, Optional, Tuple

from src import parameters
from world.simulated_world import SimulatedWorld


class Ledge(SimulatedWorld):

    def __init__(self, state: Optional[Tuple[int, ...]] = None):
        self.__size = parameters.SIZE

        if state is None:
            self.__player_id, *self.__board = list(self.reset())
        else:
            self.__player_id, *self.__board = list(state)

    def reset(self) -> Tuple[int, ...]:
        self.__player_id = 1
        self.__board = list(parameters.LEDGE_BOARD)
        return self.__get_state()

    def is_final_state(self) -> bool:
        return 2 not in self.__board

    def step(self, action: Tuple[int, int]) -> Tuple[int, ...]:
        coin_position, landing_position = action
        if landing_position >= 0:
            self.__board[landing_position], self.__board[coin_position] = self.__board[coin_position], 0
        else:
            self.__board[coin_position] = 0
        return self.__get_state()

    def generate_child_states(self) -> Tuple[Tuple[int, ...], ...]:
        possible_actions = self.__generate_possible_actions()
        return tuple(self.__generate_state(action) for action in possible_actions)

    def __get_state(self) -> Tuple[int, ...]:
        return (self.__player_id, *self.__board)

    def __generate_state(self, action: Tuple[int, int]) -> Tuple[int, ...]:
        coin_position, landing_position = action
        board = list(self.__board)
        if landing_position >= 0:
            board[landing_position], board[coin_position] = board[coin_position], 0
        else:
            board[coin_position] = 0
        return (self.__player_id, *board)

    def __generate_possible_actions(self) -> Tuple[Tuple[int, int], ...]:
        possible_actions = []
        for coin_position in range(self.__size):
            for landing_position in range(-1, coin_position):
                if self.__is_legal_action(self.__board, (coin_position, landing_position)):
                    possible_actions.append((coin_position, landing_position))
        return tuple(possible_actions)

    def __is_legal_action(self, board: List[int], action: Tuple[int, int]) -> bool:
        coin_position, landing_position = action
        if board[coin_position] == 0:
            return False
        if coin_position < landing_position:
            return False
        if coin_position == 0:
            return True
        if sum(board[landing_position:coin_position]) > 0:
            return False
        return True
