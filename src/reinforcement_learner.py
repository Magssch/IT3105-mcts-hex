import parameters
from actor import Actor
from critic.critic_factory import CriticFactory
from simulated_world import SimulatedWorld


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

    def __init__(self):
        self.__simulated_world = SimulatedWorld()
        self.__episodes = parameters.EPISODES
        self.__number_of_rollouts = parameters.NUMBER_OF_ROLLOUTS
        self.__caching_interval = parameters.ACTOR_CACHING_INTERVAL

    def __run_one_episode(self,) -> None:
        # i_s = __caching_interval  # save interval for ANET (the actor network) parameters
        # Clear Replay Buffer (RBUF)
        # Randomly initialize parameters (weights and biases) of ANET
        # for g_a in number_actualgames:
        #     (a)  Initialize the actual game board (Ba) to an empty board.
        #     (b)  s_init = startingboardstate
        #     (c)  Initialize the Monte Carlo Tree (MCT) to a singleroot, which representssinit
        #     (d)  while B_a not in a final state:
        #             Initialize Monte Carlo game board (Bmc) to same state as root.
        #             for g_s in self.__number_of_rollouts:
        #                 Use tree policy P_t to search from root to a leaf (L) of MCT. UpdateBmcwith each move.
        #                 Use ANET to choose rollout actions from L to a final state (F). UpdateBmcwith each move.
        #                 Perform MCTS backpropagation from F to root.
        #             next = g_s
        #             D = distribution of visit counts in MCT along all arcs emanating from root.
        #             Add case (root, D) to RBUF
        #             Choose actual move (a*) based on D
        #             Perform a* on root to produce successor state s*•UpdateBato s*
        #             In MCT, retain subtree rooted at s*; discard everything else.
        #             root = s*
        #     (e)  Train ANET on a random minibatch of cases from RBUF
        #     (f)  if g_a % i_s == 0:
        #             Save ANET’s current parameters for later use in tournament play.
        # next g_a
        pass

    def run(self) -> None:
        """
        Runs all episodes with pivotal parameters.
        Visualizes one round at the end.
        """
        for episode in range(self.__episodes):
            print('Episode:', episode + 1)
            self.__run_one_episode()

        print('Training completed.')
        self.__simulated_world.plot_training_data()

        if parameters.VISUALIZE_GAMES:
            print('Showing one episode with the greedy strategy.')
