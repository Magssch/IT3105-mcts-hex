from __future__ import annotations

from typing import Dict, Optional, Tuple


class TreeNode:

    def __init__(self, state: Tuple[int, ...], parent: Optional[TreeNode] = None) -> None:
        self.state = state
        self.__is_terminal: bool = False

        self.score = 0
        self.visits = 0

        self.parent = parent
        self.children: Dict[int, TreeNode] = {}

    @property
    def is_leaf(self) -> bool:
        return not bool(self.children)

    @property
    def is_terminal(self) -> bool:
        return self.__is_terminal

    @property
    def value(self) -> float:
        return self.score / self.visits if self.visits != 0 else 0

    def add_node(self, action: int, state: Tuple[int, ...]) -> TreeNode:
        child_node = TreeNode(state)
        self.children[action] = child_node
        child_node.parent = self
        return child_node

    def __eq__(self, o: TreeNode) -> bool:
        return self.state == o.state

    def __hash__(self) -> int:
        return hash(self.state)

    def __str__(self) -> str:
        return f'TreeNode(s={self.score}, v={self.visits}): {self.state}'
