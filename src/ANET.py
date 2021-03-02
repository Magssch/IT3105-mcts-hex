import random
from typing import Any, Tuple

import numpy as np
import tensorflow as tf
from keras import backend as K  # noqa
from keras.activations import softmax
from keras.layers import Dense, Input
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.utils import normalize

from src import parameters


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

    def __init__(self) -> None:
        self.__epsilon = parameters.ANET_EPSILON
        self.__learning_rate = parameters.ANET_LEARNING_RATE
        self.__activation_function = parameters.ANET_ACTIVATION_FUNCTION
        self.__optimizer = parameters.ANET_OPTIMIZER
        self.__model = self.__build_model()

    def __build_model(self) -> Sequential:
        """Builds a neural network model with the provided dimensions and learning rate"""
        input_dim, *hidden_dims, output_dim = parameters.ANET_DIMENSIONS

        model = Sequential()
        model.add(Input(shape=(input_dim,)))

        for dimension in hidden_dims:
            model.add(Dense(dimension, activation=self.__activation_function))

        model.add(Dense(output_dim, activation=softmax))

        model.compile(
            optimizer=self.__optimizer(learning_rate=self.__learning_rate),
            loss=categorical_crossentropy
        )
        model.summary()
        return model

    def choose_action(self, state: Tuple[int, ...], possible_actions: Tuple[Any, ...]) -> int:
        """Epsilon-greedy action selection function."""
        if random.random() < self.__epsilon:
            return self.choose_uniform(possible_actions)
        return self.choose_greedy(state, possible_actions)

    def choose_uniform(self, possible_actions: Tuple[Any, ...]) -> Any:
        return random.choice(possible_actions)

    def choose_greedy(self, state: Tuple[int, ...], possible_actions: Tuple[Any, ...]) -> int:
        action_probabilities = self.__model(state).tolist()
        for action in range(parameters.NUMBER_OF_ACTIONS):
            if bool(possible_actions[action]):
                action_probabilities[action] = 0
        action_probabilities = normalize(action_probabilities)
        return np.argmax(action_probabilities)

    def fit(self) -> None:
        with tf.GradientTape(persistent=True) as tape:
            target = reward + self._discount_factor * self.__values(tf.convert_to_tensor([successor_state]))  # type: ignore
            prediction = self.__values(tf.convert_to_tensor([current_state]))
            loss = self.__values.compiled_loss(target, prediction)
            td_error = target - prediction

        gradients = tape.gradient(loss, self.__values.trainable_weights)
        gradients = self.__modify_gradients(gradients, td_error)
        self.__values.optimizer.apply_gradients(zip(gradients, self.__values.trainable_weights))  # type: ignore