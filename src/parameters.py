from keras.activations import linear, relu, sigmoid, tanh  # noqa
from keras.optimizers import SGD, Adagrad, Adam, RMSprop  # noqa

from game import Game

# MCTS parameters
EPISODES = 100
NUMBER_OF_ROLLOUTS = 50 # M
UCT_C = 1.25  # "Often 1"

# Simulated World
SIZE = 3  # 3 <= k <= 10
GAME_TYPE = Game.Hex
STATE_SIZE = 1 + (SIZE ** 2 if GAME_TYPE == Game.Hex else SIZE)
NUMBER_OF_ACTIONS = SIZE ** 2 if GAME_TYPE == Game.Hex else int((SIZE ** 2 - SIZE) / 2) + 1
VISUALIZE_GAMES = False
FRAME_DELAY = 0.4
LEDGE_BOARD = (0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 2, 0, 0, 1, 1, 1)  # (0,2,0,1,0,1,1,1,0,0,0,0,1,0,1,0,1)

# ANET
ANET_EPSILON = 0
ANET_EPSILON_DECAY = 1
ANET_LEARNING_RATE = None
ANET_ACTIVATION_FUNCTION = relu  # linear, relu, sigmoid, or tanh
ANET_OPTIMIZER = Adam  # SGD, Adagrad, Adam, or RMSprop
ANET_BATCH_SIZE = 50
ANET_DIMENSIONS = (STATE_SIZE, 32, 32, NUMBER_OF_ACTIONS)

# TOPP parameters
ANETS_TO_BE_CACHED = 3
NUMBER_OF_GAMES = 10
