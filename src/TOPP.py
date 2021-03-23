import parameters
from ANET import ANET
import tensorflow as tf
from os import walk
from world.simulated_world_factory import SimulatedWorldFactory
from reinforcement_learner import ReinforcementLearner

class TOPP:

    def __init__(self) -> None:
        self.agents = self.get_agents()
        self.number_of_agents = len(self.agents)
        self.number_of_series = (self.number_of_agents * (self.number_of_agents - 1 )) / 2
        self.number_of_games = parameters.NUMBER_OF_GAMES
        self.visualize_game = parameters.VISUALIZE_GAMES

    def get_agents(self):
        _, _, models = next(walk('src/models'))
        agents = []
        for model in models:
            agent = ANET(model)
            agents.append(agent)
        return agents

    def play(self):

        for i in range(self.number_of_agents - 1):
            player_1 = self.agents[i]
            for j in range(i + 1, self.number_of_agents):
                player_2 = self.agents[j]

                for game in range(self.number_of_games):
                    print(f'p1={player_1} is playing against p2={player_2}. Round {game + 1}')
                    ReinforcementLearner.run_one_game(player_1, player_2, self.visualize_game)






if __name__ == '__main__':
    topp = TOPP()
    topp.play()
