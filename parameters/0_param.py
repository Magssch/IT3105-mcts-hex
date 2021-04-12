from keras.activations import linear, relu, sigmoid, tanh  # noqa
from keras.optimizers import SGD, Adagrad, Adam, RMSprop  # noqa

from game import Game

# RL parameters
REPLAY_BUFFER_SIZE = 32

# MCTS parameters
EPISODES = 100
SIMULATION_TIME_OUT = 0.3  # s
UCT_C = 1  # "theoretically 1"

# Simulated World
GAME_TYPE = Game.Hex
LEDGE_BOARD = (0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 2, 0, 0, 1, 1, 1)  # (0, 2, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1)
SIZE = 3 if GAME_TYPE == Game.Hex else len(LEDGE_BOARD)  # 3 <= k <= 10
STATE_SIZE = 1 + (SIZE ** 2 if GAME_TYPE == Game.Hex else SIZE)
NUMBER_OF_ACTIONS = SIZE ** 2 if GAME_TYPE == Game.Hex else int((SIZE ** 2 - SIZE) / 2) + 1
VISUALIZE_GAMES = False
FRAME_DELAY = 0.4

# ANET
ANET_EPSILON = 0
ANET_EPSILON_DECAY = 1
ANET_LEARNING_RATE = None
ANET_ACTIVATION_FUNCTION = sigmoid  # linear, relu, sigmoid, or tanh
ANET_OPTIMIZER = Adam  # SGD, Adagrad, Adam, or RMSprop
ANET_BATCH_SIZE = 32
ANET_DIMENSIONS = (STATE_SIZE, 20, 40, 20, NUMBER_OF_ACTIONS)

# TOPP parameters
ANETS_TO_BE_CACHED = 11
NUMBER_OF_GAMES = 1
