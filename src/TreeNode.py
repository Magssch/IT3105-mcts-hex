from __future__ import annotations

from typing import Optional, Tuple


class TreeNode:

    def __init__(self, state: Tuple[int], parent: Optional[TreeNode]) -> None:
        self.state = state
        self.visits = 0
        self.score = 0
        self.is_terminal = False
        self.parent = parent
        self.children = {}

    def __eq__(self, o: TreeNode) -> bool:
        return self.state == o.state

    def __hash__(self) -> int:
        return hash(self.state)
