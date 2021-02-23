from data_classes import Shape

# MCTS parameters
EPISODES = 100
NUMBER_OF_ROLLOUTS = 10
UCT_C = 1 # "Often 1" (see https://prod.liveshare.vsengsaas.visualstudio.com/join?89DF72862A01B0CD8E1C3D48F4FDE0B3500A)

# Simulated World
SIZE = 5  # 3 <= k <= 10
GAME_TYPE = 'ledge'  # ledge | hex
NUMBER_OF_STATES = SIZE ** 2
NUMBER_OF_ACTIONS = SIZE ** 2
VISUALIZE_GAMES = True
FRAME_DELAY = 0.15


# Actor-NET
ACTOR_LEARNING_RATE = 0.001
ACTOR_ACTIVATION_FUNCTION = 'relu'  # linear, sigmoid, tanh, relu
ACTOR_OPTIMIZER = 'Adagrad'  # Adagrad, Stochastic GradientDescent (SGD), RMSProp, or Adam
ACTOR_CACHING_INTERVAL = 5
ACTOR_NN_DIMENSIONS = (NUMBER_OF_STATES, 10, 30, 5, NUMBER_OF_ACTIONS)

# TOPP parameters
NUMBER_OF_GAMES = 4
