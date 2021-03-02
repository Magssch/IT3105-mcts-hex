from abc import ABC, abstractmethod
from typing import Tuple


class SimulatedWorld(ABC):

    @abstractmethod
    def reset(self) -> Tuple[int, ...]:
        raise NotImplementedError

    @abstractmethod
    def generate_child_states(self, state: Tuple[int, ...]) -> Tuple[Tuple[int, ...]]:
        raise NotImplementedError

    @abstractmethod
    def get_legal_actions(self, state: Tuple[int, ...]) -> Tuple[bool, ...]:
        raise NotImplementedError

    @abstractmethod
    def is_final_state(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Tuple[int, ...]) -> Tuple[int, ...]:
        raise NotImplementedError
