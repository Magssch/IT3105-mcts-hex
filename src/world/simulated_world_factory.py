from typing import Optional, Tuple
from .simulated_world import SimulatedWorld
from .ledge import Ledge
from .hex import Hex
import parameters
from game import Game


class SimulatedWorldFactory:

    @staticmethod
    def get_simulated_world(state: Optional[Tuple[int, ...]] = None) -> SimulatedWorld:
        if parameters.GAME_TYPE == Game.Ledge:
            return Ledge()
        else:
            return Hex()
