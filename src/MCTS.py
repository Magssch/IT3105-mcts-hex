from typing import Callable, Tuple

import parameters
from TreeNode import TreeNode
from world.simulated_world import SimulatedWorld

Policy = Callable[[Tuple[int, ...], Tuple[int, ...]], int]  # (s, valid_actions) -> a


class MCTS:

    def __init__(self, initial_state: Tuple[int, ...]) -> None:
        self.root = TreeNode(initial_state)
        self.action_space = parameters.NUMBER_OF_ACTIONS

    def update_root(self, action: int) -> None:
        self.root = self.root.children[action]
        self.root.parent = None

    def get_normalized_distribution(self) -> Tuple[float, ...]:
        # print(list(map(lambda node: node.visits, self.root.children.values())), "/", self.root.visits - 1)
        distribution = []
        for action in range(self.action_space):
            if action in self.root.children:
                distribution.append(float(self.root.children[action].visits) / float(self.root.visits - 1))
            else:
                distribution.append(0.0)
        return tuple(distribution)

    def do_one_simulation(self, default_policy: Policy, world: SimulatedWorld) -> None:
        # Tree search
        current_node = self.root
        while current_node.is_not_leaf:
            action = current_node.tree_policy()
            # print('UCT values', list(map(lambda key: (key, self.UCT(current_node.children[key])), current_node.children.keys())))
            # print(f'Node chosen {action}. For player {current_node.player_id}')
            world.step(action)
            current_node = current_node.children[action]

        # Node expansion
        if not world.is_final_state() and current_node.visits != 0:
            for action, legal in enumerate(world.get_legal_actions()):
                if bool(legal):
                    current_node.add_node(action, world.generate_state(action))
            current_node = list(current_node.children.values())[0]

        # Rollout
        current_state = current_node.state
        while not world.is_final_state():
            legal_actions = world.get_legal_actions()
            action = default_policy(current_state, legal_actions)
            current_state, winner = world.step(action)

        # Backpropagation
        while current_node is not None:
            current_node.add_reward(world.get_winner_id())
            current_node.increment_visit_count()
            current_node = current_node.parent
