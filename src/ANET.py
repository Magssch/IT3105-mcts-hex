import random
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from keras import backend as K  # noqa
from keras.activations import softmax
from keras.layers import Dense, Input
from keras.losses import kl_divergence
from keras.models import Sequential
from keras.utils import normalize

import parameters


class ANET:
    """
    Table-based Actor using the epsilon-greedy strategy

    ...

    Attributes
    ----------

    Methods
    -------
    choose_action(state, possible_actions):

        Epsilon-greedy action selection function.
    update(td_error):
        Updates the policy function, then eligibilities for each state-action
        pair in the episode based on the td_error from the critic.
        Also decays the epsilon based on the epsilon decay rate.
    reset_eligibilities():
        Sets all eligibilities to 0.0
    replace_eligibilities(state, action):
        Replaces trace e(state) with 1.0
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.__epsilon = parameters.ANET_EPSILON
        self.__epsilon_decay_rate = parameters.ANET_EPSILON_DECAY

        self.__learning_rate = parameters.ANET_LEARNING_RATE
        self.__activation_function = parameters.ANET_ACTIVATION_FUNCTION
        self.__optimizer = parameters.ANET_OPTIMIZER

        if model_name is None:
            self.__model: Sequential = self.__build_model()
        else:
            self.load(model_name)

        self.__loss_history = []
        self.__epsilon_history = []

    def __build_model(self) -> Sequential:
        """Builds a neural network model with the provided dimensions and learning rate"""
        self.__name = 'Reidar'
        input_dim, *hidden_dims, output_dim = parameters.ANET_DIMENSIONS

        model = Sequential()
        model.add(Input(shape=(input_dim,)))

        for dimension in hidden_dims:
            model.add(Dense(dimension, activation=self.__activation_function))

        model.add(Dense(output_dim, activation=softmax))

        model.compile(
            optimizer=(self.__optimizer(learning_rate=self.__learning_rate) if self.__learning_rate is not None else self.__optimizer()),
            loss=kl_divergence
        )
        model.summary()
        return model

    def save(self, model_name: str) -> None:
        self.__model.save(f'models/{model_name}')

    def load(self, model_name: str) -> None:
        self.__name = 'Agent-e' + model_name.replace('.h5', '')
        self.__model = tf.keras.models.load_model(f'models/{model_name}')

    def set_epsilon(self, epsilon: float):
        self.__epsilon = epsilon

    def choose_action(self, state: Tuple[int, ...], valid_actions: Tuple[int, ...]) -> int:
        """Epsilon-greedy action selection function."""
        if random.random() < self.__epsilon:
            return self.choose_uniform(valid_actions)
        return self.choose_greedy(state, valid_actions)

    def choose_uniform(self, valid_actions: Tuple[int, ...]) -> int:
        assert sum(valid_actions) > 0, 'Illegal argument, valid actions cannot be empty'
        return random.choice([i for i, action in enumerate(valid_actions) if action == 1])

    def choose_greedy(self, state: Tuple[int, ...], valid_actions: Tuple[int, ...]) -> int:
        action_probabilities = self.__model(tf.convert_to_tensor([state])).numpy()
        action_probabilities = action_probabilities * np.array(valid_actions)
        action_probabilities = normalize(action_probabilities)
        return np.argmax(action_probabilities)

    def fit(self, batch: np.ndarray) -> None:
        X, Y = batch[:, :parameters.STATE_SIZE], batch[:, parameters.STATE_SIZE:]
        history = self.__model.fit(X, Y, batch_size=parameters.ANET_BATCH_SIZE)

        # Used for visualization
        self.__loss_history.append(history.history["loss"][0])
        self.__epsilon_history.append(self.__epsilon)

        self.__epsilon *= self.__epsilon_decay_rate  # decay epislon

    @property
    def loss_history(self):
        return self.__loss_history

    @property
    def epsilon_history(self):
        return self.__epsilon_history

    def __str__(self) -> str:
        return self.__name

    def __repr__(self) -> str:
        return self.__name
