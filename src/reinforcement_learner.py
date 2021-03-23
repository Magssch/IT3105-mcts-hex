import random
from visualize import Visualize

import numpy as np

import parameters
from ANET import ANET
from MCTS import MCTS
from world.simulated_world_factory import SimulatedWorldFactory


class ReinforcementLearner:
    """
    Reinforcement Learner agent using the Actor-Critic architecture

    ...

    Attributes
    ----------

    Methods
    -------
    run():
        Runs all episodes with pivotal parameters
    """

    def __init__(self) -> None:
        self.__actual_game = SimulatedWorldFactory.get_simulated_world()
        self.__replay_buffer = np.empty((0, parameters.STATE_SIZE + parameters.NUMBER_OF_ACTIONS))  # RBUF
        self.__ANET = ANET()

        self.__episodes = parameters.EPISODES
        self.__number_of_rollouts = parameters.NUMBER_OF_ROLLOUTS
        self.__caching_interval = parameters.ANET_CACHING_INTERVAL

    def __run_one_episode(self,) -> None:
        initial_game_state = self.__actual_game.reset()
        monte_carlo_tree = MCTS(initial_game_state)
        root_state = initial_game_state

        while not self.__actual_game.is_final_state():
            monte_carlo_game = SimulatedWorldFactory.get_simulated_world(root_state)

            for _ in range(self.__number_of_rollouts):
                leaf_node = monte_carlo_tree.tree_search(monte_carlo_tree.root, monte_carlo_game)
                winner = monte_carlo_tree.do_rollout(leaf_node, self.__ANET.choose_action, monte_carlo_game)
                monte_carlo_tree.do_backpropagation(leaf_node, winner)
                monte_carlo_game.reset(root_state)

            target_distribution = monte_carlo_tree.get_normalized_distribution()
            self.__replay_buffer = np.append(self.__replay_buffer, np.array([root_state + target_distribution]), axis=0)

            legal_actions = self.__actual_game.get_legal_actions()
            action = self.__ANET.choose_greedy(root_state, legal_actions)
            next_state, _ = self.__actual_game.step(action)

            monte_carlo_tree.update_root(action)
            root_state = next_state

        # Train ANET on a random minibatch of cases from RBUF
        random_rows = random.sample(range(0, self.__replay_buffer.shape[0]), 10)
        self.__ANET.fit(self.__replay_buffer[random_rows])

    def run(self) -> None:
        """
        Runs all episodes with pivotal parameters.
        Visualizes one round at the end.
        """
        # i_s = self.__caching_interval  # save interval for ANET (the actor network) parameters
        # Clear Replay Buffer (RBUF)
        # Randomly initialize parameters (weights and biases) of ANET
        for episode in range(self.__episodes):
            print('Episode:', episode + 1)
            self.__run_one_episode()

            if episode % self.__caching_interval == 0:
                # Save ANET’s current parameters for later use in tournament play.
                self.__ANET.save(str(episode) + ".h5")

        if parameters.VISUALIZE_GAMES:
            print('Showing one episode with the greedy strategy.')
            ReinforcementLearner.run_one_game(self.__ANET, self.__ANET, True)

    @staticmethod
    def run_one_game(player_1: ANET, player_2: ANET, visualize=False):
        world = SimulatedWorldFactory.get_simulated_world()
        current_state = world.reset()

        if visualize:
            Visualize.initialize_board(current_state)

        winner = 0
        while not world.is_final_state():
            player_id = current_state[0]
            legal_actions = world.get_legal_actions()

            player = player_1 if player_id == 1 else player_2
            action = player.choose_action(current_state, legal_actions)

            current_state, winner = world.step(action)

            if visualize:
                Visualize.draw_board(current_state)

        print(f'Player {winner} has won the game.')
