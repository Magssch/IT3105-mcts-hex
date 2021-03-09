from math import log, sqrt
from typing import Callable, Tuple

from src import parameters
from TreeNode import TreeNode
from world.simulated_world import SimulatedWorld

Policy = Callable[[Tuple[int, ...], Tuple[int, ...]], int]  # s -> a


class MCTS:

    def __init__(self, initial_state: Tuple[int, ...]) -> None:
        self.root = TreeNode(initial_state)
        self.action_space = parameters.NUMBER_OF_ACTIONS

    def update_root(self, action: int) -> None:
        self.root.parent = None
        self.root = self.root.children[action]

    def get_normalized_distribution(self) -> Tuple[float, ...]:
        distribution = []
        for action in range(self.action_space):
            if action in self.root.children:
                distribution[action] = self.root.children[action].visits / self.root.visits
            else:
                distribution[action] = 0
        return tuple(distribution)

    def tree_search(self, rootNode: TreeNode, world: SimulatedWorld) -> TreeNode:
        current_node = rootNode
        while not current_node.is_leaf:
            action = self.tree_policy(current_node)
            world.step(action)
            current_node = current_node.children[action]

        # Node expansion
        if current_node.visits != 0:
            for action in world.get_legal_actions(current_node.state):
                current_node.add_node(action, world.generate_state(action))  # ??
            current_node = list(current_node.children.values())[0]

        return current_node

    def do_rollout(self, leaf_node: TreeNode, default_policy: Policy, world: SimulatedWorld) -> int:
        current_state = leaf_node.state
        reward = 0
        while not world.is_final_state():
            legal_actions = world.get_legal_actions(current_state)
            action = default_policy(current_state, legal_actions)
            current_state, reward = world.step(action)
        return reward

    def do_backpropagation(self, leaf_node: TreeNode, reward: int) -> None:
        current_node = leaf_node
        while current_node is not None:
            current_node.score += reward
            current_node.visits += 1
            current_node = current_node.parent

    def tree_policy(self, node: TreeNode) -> int:
        """
        Choses an action based on the UCT score of that corresponding node
        """
        return max(node.children.keys(), key=lambda key: self.UCT(node.children[key]))

    def UCT(self, node: TreeNode) -> float:
        return node.value + parameters.UCT_C * sqrt(2 * log(self.root.visits) / node.visits + 1)
