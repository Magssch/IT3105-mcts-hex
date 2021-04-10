from __future__ import annotations

from math import log, sqrt
from typing import Dict, Optional, Tuple

import parameters


class TreeNode:

    player_reward = {
        1: 1,
        2: -1
    }

    def __init__(self, state: Tuple[int, ...], parent: Optional[TreeNode] = None) -> None:
        self.state = state

        self.score = 0
        self.visits = 0

        self.c = -parameters.UCT_C if state[0] == 1 else parameters.UCT_C
        self.parent = parent
        self.children: Dict[int, TreeNode] = {}

    @property
    def UCT(self) -> float:
        if self.visits == 0:
            return self.c * float("inf")
        exploitation = self.score / self.visits
        exploration = self.c * sqrt(2 * log(self.parent.visits) / (self.visits))
        return exploitation + exploration

    def tree_policy(self) -> int:
        policy_function = max if self.state[0] == 1 else min
        return policy_function(self.children.keys(), key=lambda key: self.children[key].UCT)

    @property
    def is_not_leaf(self) -> bool:
        return bool(self.children)

    def add_reward(self, winner: int) -> None:
        self.score += TreeNode.player_reward[winner]

    def increment_visit_count(self) -> None:
        self.visits += 1

    def add_node(self, action: int, state: Tuple[int, ...]) -> TreeNode:
        child_node = TreeNode(state, self)
        self.children[action] = child_node
        return child_node

    def eq(self, o: TreeNode) -> bool:
        return self.state == o.state

    def hash(self) -> int:
        return hash(self.state)
