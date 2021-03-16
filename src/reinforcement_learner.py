import random

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
        self.__actual_game = SimulatedWorldFactory.get_simulated_world()  # (a) =B_a
        self.__replay_buffer = np.empty((0, 1 + parameters.NUMBER_OF_STATES + parameters.NUMBER_OF_ACTIONS))  # RBUF
        self.__ANET = ANET()

        self.__episodes = parameters.EPISODES
        self.__number_of_rollouts = parameters.NUMBER_OF_ROLLOUTS
        self.__caching_interval = parameters.ANET_CACHING_INTERVAL

    def __run_one_episode(self,) -> None:
        initial_game_state = self.__actual_game.reset()  # (b)
        monte_carlo_tree = MCTS(initial_game_state)  # (c)
        root_state = initial_game_state

        while not self.__actual_game.is_final_state():  # (d)
            # monte_carlo_board = B_mc
            monte_carlo_game = SimulatedWorldFactory.get_simulated_world(root_state)  # (d.1)

            for search_game in range(self.__number_of_rollouts):  # (d.2) search_game brukes ikke til noe
                print("Search game", search_game)
                leaf_node = monte_carlo_tree.tree_search(monte_carlo_tree.root, monte_carlo_game)  # (d.3) tree_policy (UCB1 / UCT)
                reward = monte_carlo_tree.do_rollout(leaf_node, self.__ANET.choose_action, monte_carlo_game)  # (d.4)
                monte_carlo_tree.do_backpropagation(leaf_node, reward)  # (d.5)
                monte_carlo_game.reset(root_state)

            print(self.__replay_buffer)
            target_distribution = monte_carlo_tree.get_normalized_distribution()  # (d.6) ??
            self.__replay_buffer = np.append(self.__replay_buffer, np.array([root_state + target_distribution]), axis=0)  # (d.7)
            print(self.__replay_buffer)

            legal_actions = self.__actual_game.get_legal_actions()
            action = self.__ANET.choose_greedy(root_state, legal_actions)  # (d.8) argmax based on softmax
            next_state, reward = self.__actual_game.step(action)  # (d.9)

            monte_carlo_tree.update_root(action)
            root_state = next_state

        # Train ANET on a random minibatch of cases from RBUF
        random_rows = random.sample(range(0, self.__replay_buffer.shape[0]), 10)
        self.__ANET.fit(self.__replay_buffer[random_rows])  # (e)


        # (a)  Initialize the actual game board (B_a) to an empty board.
        # (b)  s_init = startingboardstate
        # (c)  Initialize the Monte Carlo Tree (MCT) to a single root, which represents s_init.
        # (d)  while B_a not in a final_state:
        #         Initialize Monte Carlo game board (B_mc) to same state as root.
        #         for g_s in self.__number_of_rollouts:
        #             Use tree policy P_t to search from root to a leaf (L) of MCT. Update B_mc with each move.
        #             Use ANET to choose rollout actions from L to a final state (F). Update B_mc with each move.
        #             Perform MCTS backpropagation from F to root.
        #         next = g_s
        #         D = distribution of visit counts in MCT along all arcs emanating from root.
        #         Add case (root, D) to RBUF
        #         Choose actual move (a*) based on D
        #         Perform a* on root to produce successor state s*
        #         Update B_a to s*
        #         In MCT, retain subtree rooted at s*; discard everything else.
        #         root = s*
        # (e)  Train ANET on a random minibatch of cases from RBUF


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
                pass

        if parameters.VISUALIZE_GAMES:
            print('Showing one episode with the greedy strategy.')
