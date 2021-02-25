from typing import Optional, Tuple

from src import parameters
from world.simulated_world import SimulatedWorld


class Ledge(SimulatedWorld):

    def __init__(self, state: Optional[Tuple[int, ...]]):
        self.__size = parameters.SIZE
