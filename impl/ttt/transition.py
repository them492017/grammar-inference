from __future__ import annotations
from typing import Optional, TYPE_CHECKING

from ttt.print import print

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
    hypothesis: Hypothesis
    aseq: str
    _target_node: Node
    _target_state: Optional[State]

    def __init__(
        self,
        hypothesis: Hypothesis,
        aseq: str,
        target: Node
    ) -> None:
        self.hypothesis = hypothesis
        self.aseq = aseq
        self._target_node = target
        target.incoming_non_tree.add(self)
        self._target_state = None

    def __repr__(self) -> str:
        return f"Transition<aseq='{self.aseq}', is_tree: {self.is_tree()}>"

    def is_tree(self) -> bool:
        return self._target_state is not None
    
    @property
    def target_node(self) -> Node:
        if self._target_state:
            # is tree transition
            return self._target_state.node
        return self._target_node

    @target_node.setter
    def target_node(self, tgt: Node):
        assert not self.is_tree()
        print(f"Changing target node for transition with aseq {self.aseq} from {self.target_node} to {tgt}")

        # update incoming transitions
        self.target_node.incoming_non_tree.remove(self)
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
        assert not self.is_tree()
        self._target_state = state

    def make_tree(self, node: Node) -> State:
        if self.is_tree():
            assert self._target_state
            state = self._target_state
            raise ValueError("transition is already a tree transition")
        else:
            state = self.hypothesis.add_state(self.aseq)
            self.target_node.incoming_non_tree.remove(self)
            self.target_state = state

            # if the node corresponding to this state is in the 1 subtree of the root,
            # it should be final
            if ("", True) in node.signature:
                self.hypothesis.make_final(state)

        node.link(state)

        return state
