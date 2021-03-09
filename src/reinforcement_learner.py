import random

from MCT import MCT

import parameters
from world import SimulatedWorld


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
        self.__simulated_world = SimulatedWorld()  # (a) =B_a
        self.__replay_buffer = []  # RBUF
        self.__ANET = ANET()

        self.__episodes = parameters.EPISODES
        self.__number_of_rollouts = parameters.NUMBER_OF_ROLLOUTS
        self.__caching_interval = parameters.ANET_CACHING_INTERVAL

    def __run_one_episode(self,) -> None:
        root = self.__simulated_world.reset()  # (b)
        monte_carlo_tree = MCT(root)  # (c)

        while not self.__simulated_world.is_final_state():  # (d)
            B_mc = SimulatedWorld(root)  # (d.1)

            for search_game in range(self.__number_of_rollouts):  # (d.2) search_game brukes ikke til noe
                leaf_node = monte_carlo_tree.tree_search(root)  # (d.3) tree_policy (UCB1 / UCT)
                # TODO: Node expansion!
                final_node = monte_carlo_tree.do_rollout(leaf_node, self.__ANET)  # (d.4)
                monte_carlo_tree.do_backpropagation(final_node)  # (d.5)

            D = monte_carlo_tree.get_normalized_distribution()  # (d.6) ??
            self.__replay_buffer.append((root, D))  # (d.7)
            action = self.__ANET.choose_action((root, D))  # (d.8) argmax based on softmax
            next_state = self.__simulated_world.step(action)  # (d.9)
            monte_carlo_tree.set_root(next_state)
            root = next_state

        self.__ANET.fit(random.choices(self.__replay_buffer, k=10))  # (e)

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
        pass

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
