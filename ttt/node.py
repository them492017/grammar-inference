from __future__ import annotations
from typing import Optional, Any, Protocol, Self


class Node(Protocol):
    """
    A node in the discrimination tree
    """


    def __init__(self, children: tuple[Optional[Node], Optional[Node]]) -> None:
        self.children = children

    @classmethod
    def make_leaf(cls, _arg: Any) -> LeafNode:
        return cls((None, None))

    @classmethod
    def make_inner(cls, children: t) -> InnerNode:
        return cls((None, None))


class LeafNode(Node):
    ...


class InnerNode(Node):
    children: tuple[Optional[Node], Optional[Node]]
