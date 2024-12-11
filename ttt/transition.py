from typing import Protocol, Union, Optional

from new_node import Node
from state import State


class Transition(Protocol):
    """
    A transition (either a tree transition or a non-tree transition) in the
    spanning-tree hypothesis
    """
    aseq: str
    
    @property
    def target(self) -> Optional[Union[Node, State]]:
        ...


class NonTreeTransition(Transition):
    tgt: Optional[Node]

    def __init__(self, aseq: str, tgt: Optional[Node]) -> None:
        self.aseq = aseq
        self.tgt = tgt

    @property
    def target(self) -> Optional[Node]:
        return self.tgt

    def make_tree(self) -> None:
        ...


class TreeTransition(Transition):
    tgt: State

    def __init__(self, aseq: str, tgt: State) -> None:
        self.aseq = aseq
        self.tgt = tgt

    @property
    def target(self) -> State:
        return self.tgt
