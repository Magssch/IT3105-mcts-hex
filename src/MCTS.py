from typing import Tuple

import TreeNode


class MCTS:

    def search(self, initial_state: Tuple[int]):
        self.root = TreeNode(initial_state)
        # ...

    def do_rollout(self, node: TreeNode) -> int:
        pass

    def do_backpropagation(self, node: TreeNode, score: int) -> None:
        pass
