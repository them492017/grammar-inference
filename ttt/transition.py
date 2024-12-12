from __future__ import annotations
from typing import Union, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from node import Node
    from state import Hypothesis, State


class Transition:
    """
    A transition (either a tree transition or a non-tree transition) in the
    spanning-tree hypothesis
    """
    is_tree: bool
    hypothesis: Hypothesis
    aseq: str
    _target: Union[Node, Optional[State]]

    def __init__(
        self,
        is_tree: bool,
        hypothesis: Hypothesis,
        aseq: str,
        target: Optional[Union[Node, State]]
    ) -> None:
        self.is_tree = is_tree
        self.hypothesis = hypothesis
        self.aseq = aseq
        self._target = target
    
    @property
    def target(self) -> Optional[Union[Node, State]]:
        if self._target is None:
            return None
        if self.is_tree:
            # assert isinstance(self._target, State)
            return self._target
        else:
            # assert isinstance(self._target, Node)
            return self._target

    @target.setter
    def target(self, tgt: Node):
        self._target = tgt

    def make_tree(self) -> State:
        assert not self.is_tree
        self.is_tree = True
        state = self.hypothesis.add_state(self.aseq)
        self._target = state
        return state
