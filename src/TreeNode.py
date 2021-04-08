from __future__ import annotations

from typing import Dict, Optional, Tuple


class TreeNode:

    player_reward = {
        1: 1,
        2: -1
    }

    def __init__(self, state: Tuple[int, ...], parent: Optional[TreeNode] = None) -> None:
        self.__state = state
        self.__is_terminal: bool = False

        self.__score = 0
        self.__visits = 0

        self.parent = parent
        self.children: Dict[int, TreeNode] = {}

    @property
    def state(self) -> Tuple[int, ...]:
        return self.__state

    @property
    def visits(self) -> int:
        return self.__visits

    @property
    def player_id(self) -> int:
        return self.__state[0]

    @property
    def is_leaf(self) -> bool:
        return not bool(self.children)

    @property
    def is_terminal(self) -> bool:
        return self.__is_terminal

    @property
    def value(self) -> float:
        return self.__score / self.__visits if self.__visits != 0 else 0

    def set_terminal(self):
        self.__is_terminal = True

    def get_parent(self) -> Optional[TreeNode]:
        return self.parent

    def add_reward(self, winner: int) -> None:
        self.__score += TreeNode.player_reward[winner]

    def increment_visit_count(self) -> None:
        self.__visits += 1

    def add_node(self, action: int, state: Tuple[int, ...]) -> TreeNode:
        child_node = TreeNode(state, self)
        self.children[action] = child_node
        return child_node

    def __eq__(self, o: TreeNode) -> bool:
        return self.__state == o.__state

    def __hash__(self) -> int:
        return hash(self.__state)

    def __str__(self) -> str:
        return f'TreeNode(s={self.__score}, v={self.visits}): {self.__state}'
