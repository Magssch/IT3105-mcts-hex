from math import log, sqrt
from typing import Callable, Tuple

import numpy as np

from src import parameters
from TreeNode import TreeNode
from world.simulated_world import SimulatedWorld

Policy = Callable[[Tuple[int, ...], Tuple[bool, ...]], int]  # s -> a


class MCTS:

    def __init__(self, initial_state: Tuple[int, ...]) -> None:
        self.root = TreeNode(initial_state)
        self.action_space = parameters.NUMBER_OF_ACTIONS

    def set_root(self, node: TreeNode) -> None:
        node.parent = None
        self.root = node

    def get_normalized_distribution(self) -> Tuple[float, ...]:
        distribution = []
        for action in range(self.action_space):
            if action in self.root.children:
                distribution[action] = self.root.children[action].visits / self.root.visits
            else:
                distribution[action] = 0
        return tuple(distribution)

    def tree_search(self, node: TreeNode, world: SimulatedWorld) -> TreeNode:
        current_node = node
        while not current_node.is_leaf:
            current_node = self.tree_policy(current_node)  # <- tree policy
        return current_node

    def do_rollout(self, leafNode: TreeNode, default_policy: Policy, world: SimulatedWorld) -> TreeNode:
        current_node = leafNode
        while not current_node.is_terminal:
            legal_moves = world.get_legal_actions(current_node.state)
            action = default_policy(current_node.state, legal_moves)
            next_state = world.step(action)
            current_node = current_node.add_node(action, next_state)
        return current_node

    def do_backpropagation(self, terminalNode: TreeNode, score: int) -> None:
        child = terminalNode
        parent = child.parent
        while parent is not None:
            parent.score += child.score
            parent.visits += 1
            parent = parent.parent

    def add_child_node(self, node: TreeNode) -> None:
        pass

    def tree_policy(self, node: TreeNode) -> TreeNode:
        """
        Choses a child node based on the UCT score
        """
        return max(node.children.values(), key=self.UCT)

    def UCT(self, node: TreeNode) -> float:
        return node.value + parameters.UCT_C * sqrt(2 * log(self.root.visits) / node.visits + 1)
