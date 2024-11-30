from typing import Protocol, Union

from node import Node
from state import State


class Transition(Protocol):
    """
    A transition (either a tree transition or a non-tree transition) in the
    spanning-tree hypothesis
    """
    aseq: str
    
    @property
    def target(self) -> Union[Node, State]:
        ...


class NonTreeTransition(Transition):
    tgt: Node

    def __init__(self, aseq: str, tgt: Node) -> None:
        self.aseq = aseq
        self.tgt = tgt

    @property
    def target(self) -> Node:
        return self.tgt


class TreeTransition(Transition):
    tgt: State

    def __init__(self, aseq: str, tgt: State) -> None:
        self.aseq = aseq
        self.tgt = tgt

    @property
    def target(self) -> State:
        return self.tgt
