from keras.activations import linear, relu, sigmoid, tanh  # noqa
from keras.optimizers import SGD, Adagrad, Adam, RMSprop  # noqa

from game import Game

# MCTS parameters
EPISODES = 100
NUMBER_OF_ROLLOUTS = 10
UCT_C = 1  # "Often 1"

# Simulated World
SIZE = 5  # 3 <= k <= 10
GAME_TYPE = Game.Hex
STATE_SIZE = 1 + (SIZE ** 2 if GAME_TYPE == Game.Hex else SIZE)
NUMBER_OF_ACTIONS = SIZE ** 2 if GAME_TYPE == Game.Hex else int((SIZE ** 2 - SIZE) / 2) + 1
VISUALIZE_GAMES = False
FRAME_DELAY = 0.4
LEDGE_BOARD = (0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 2, 0, 0, 1, 1, 1)  # (0,2,0,1,0,1,1,1,0,0,0,0,1,0,1,0,1)

# ANET
ANET_EPSILON = 0.5
ANET_LEARNING_RATE = None
ANET_ACTIVATION_FUNCTION = relu  # linear, relu, sigmoid, or tanh
ANET_OPTIMIZER = Adam  # SGD, Adagrad, Adam, or RMSprop
ANET_BATCH_SIZE = 15
ANET_DIMENSIONS = (STATE_SIZE, 32, 32, NUMBER_OF_ACTIONS)

# TOPP parameters
ANETS_TO_BE_CACHED = 5
NUMBER_OF_GAMES = 1
