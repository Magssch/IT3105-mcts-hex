from __future__ import annotations

from typing import Optional, Tuple


class TreeNode:

    def __init__(self, state: Tuple[int], parent: Optional[TreeNode]) -> None:
        self.state = state
        self.is_terminal = False

        self.score = 0
        self.visit_count = 0

        self.parent = parent
        self.children = {}

    @property
    def value(self):
        return self.score / self.visit_count

    def __eq__(self, o: TreeNode) -> bool:
        return self.state == o.state

    def __hash__(self) -> int:
        return hash(self.state)
