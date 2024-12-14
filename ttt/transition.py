from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from node import Node
    from state import Hypothesis, State


class Transition:
    """
    A transition (either a tree transition or a non-tree transition) in the
    spanning-tree hypothesis.

    Tree transitions point to states in the spanning tree hypothesis, and
    their targets are final.

    Non-tree transitions point to nodes in the discrimination tree. A non-tree
    transition is called closed if the node has an associated state (must be a leaf),
    and open otherwise.
    """
    is_tree: bool
    hypothesis: Hypothesis
    aseq: str
    _target_node: Node
    _target_state: Optional[State]

    def __init__(
        self,
        is_tree: bool,
        hypothesis: Hypothesis,
        aseq: str,
        target: Node
    ) -> None:
        self.is_tree = is_tree
        self.hypothesis = hypothesis
        self.aseq = aseq
        self.target_node = target
        self._target_state = None
    
    @property
    def target_node(self) -> Node:
        return self._target_node

    @target_node.setter
    def target_node(self, tgt: Node):
        # TODO: had to remove since make_tree() doesnt set target node
        # assert not self.is_tree
        tgt.incoming_non_tree.add(self)
        self._target_node = tgt

    @property
    def target_state(self) -> Optional[State]:
        if self._target_state is None:
            return self._target_node.state
        else:
            return self._target_state

    @target_state.setter
    def target_state(self, state: State) -> Optional[State]:
        assert not self.is_tree
        self.is_tree = True
        self._target_state = state

    def make_tree(self, node: Node) -> State:
        assert not self.is_tree
        state = self.hypothesis.add_state(self.aseq)
        self.target_state = state

        # if the node corresponding to this state is in the 1 subtree of the root,
        # it should be final
        if ("", True) in node.signature:
            self.hypothesis.make_final(state)

        print("incoming", node.incoming_non_tree)
        self.hypothesis.open_transitions += list(node.incoming_non_tree)

        return state
