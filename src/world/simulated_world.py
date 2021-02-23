from abc import ABC, abstractmethod
from typing import Tuple


class SimulatedWorld(ABC):

    @abstractmethod
    def reset() -> Tuple[int]:
        raise NotImplementedError

    @abstractmethod
    def generate_child_states(state: Tuple[int]) -> Tuple[Tuple[int]]:
        raise NotImplementedError

    @abstractmethod
    def is_final_state(state: Tuple[int]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def step(action) -> Tuple[int]:
        raise NotImplementedError
