from keras.activations import linear, relu, sigmoid, tanh  # noqa
from keras.optimizers import SGD, Adagrad, Adam, RMSprop  # noqa

from game import Game

# RL parameters
REPLAY_BUFFER_SIZE = 64

# MCTS parameters
EPISODES = 100
SIMULATION_TIME_OUT = 0.25 # s
UCT_C = 1  # "Often 1"

# Simulated World
SIZE = 4  # 3 <= k <= 10
GAME_TYPE = Game.Hex
STATE_SIZE = 1 + (SIZE ** 2 if GAME_TYPE == Game.Hex else SIZE)
NUMBER_OF_ACTIONS = SIZE ** 2 if GAME_TYPE == Game.Hex else int((SIZE ** 2 - SIZE) / 2) + 1
LEDGE_BOARD = (0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 2, 0, 0, 1, 1, 1)  # (0,2,0,1,0,1,1,1,0,0,0,0,1,0,1,0,1)
VISUALIZE_GAMES = False
FRAME_DELAY = 0.4

# ANET
ANET_EPSILON = 0
ANET_EPSILON_DECAY = 1
ANET_LEARNING_RATE = 0.001
ANET_ACTIVATION_FUNCTION = relu  # linear, relu, sigmoid, or tanh
ANET_OPTIMIZER = Adam  # SGD, Adagrad, Adam, or RMSprop
ANET_BATCH_SIZE = 64
ANET_DIMENSIONS = (STATE_SIZE, 36, 36, NUMBER_OF_ACTIONS)

# TOPP parameters
ANETS_TO_BE_CACHED = 11
NUMBER_OF_GAMES = 10
