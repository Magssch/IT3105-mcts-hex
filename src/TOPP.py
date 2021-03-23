from os import walk

import tensorflow as tf

import parameters
from ANET import ANET
from reinforcement_learner import ReinforcementLearner
from world.simulated_world_factory import SimulatedWorldFactory


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
        for model in sorted(models, key=lambda name: int(name.split(".")[0])):
            agent = ANET(model)
            agents.append(agent)
        return agents

    def run(self):
        statisitcs = [0 for _ in range(self.number_of_agents)]
        for i in range(self.number_of_agents - 1):
            player_1 = self.agents[i]
            for j in range(i + 1, self.number_of_agents):
                player_2 = self.agents[j]

                for game in range(self.number_of_games):
                    print(f'p1={player_1} is playing against p2={player_2}. Round {game + 1}')
                    winner = ReinforcementLearner.run_one_game(player_1, player_2, self.visualize_game)
                    if winner == 1:
                        statisitcs[i] += 1
                    else:
                        statisitcs[j] += 1
        self.plot_statistics(statisitcs)

    def plot_statistics(self, statisitcs):
        for agent, wins in zip(self.agents, statisitcs):
            print(f'{str(agent):>10} has won {wins}/{self.number_of_games * (self.number_of_agents - 1)} games')








if __name__ == '__main__':
    topp = TOPP()
    topp.play()
