from __future__ import annotations
from typing import Optional, Protocol

from teacher import Teacher
from state import State


class Node(Protocol):
    """
    A node in the discrimination tree
    """
    children: tuple[Optional[Node], Optional[Node]]
    parent_info = Optional[tuple[InnerNode, bool]]
    
    def __init__(self, children: tuple[Optional[Node], Optional[Node]]) -> None:
        self.children = children
        self.parent_info = None

    @classmethod
    def make_leaf(cls) -> LeafNode:
        return LeafNode((None, None))

    @classmethod
    def make_inner(
        cls,
        discriminator: str,
        children: tuple[Optional[Node], Optional[Node]] = (None, None),
    ) -> InnerNode:
        return InnerNode(children, discriminator)

    @property
    def signature(self) -> list[tuple[str, bool]]:  # bool vs int?
        if self.parent_info is None:
            return []

        return [
            (self.parent_info[0].discriminator, self.parent_info[1]),
            *self.parent_info[0].signature
        ]

    def sift(self, s: str, teacher: Teacher) -> LeafNode:
        ...


class LeafNode(Node):
    state: Optional[State]

    def __init__(self, children: tuple[Optional[Node], Optional[Node]]) -> None:
        super().__init__(children)
        self.state = None

    def sift(self, s: str, teacher: Teacher) -> LeafNode:
        return self


class InnerNode(Node):
    children: tuple[Optional[Node], Optional[Node]]
    discriminator: str

    def __init__(self, children: tuple[Optional[Node], Optional[Node]],
                 discriminator: str) -> None:
        super().__init__(children)
        self.discriminator = discriminator

    def sift(self, s: str, teacher: Teacher) -> LeafNode:
        if teacher.is_member(s):
            if self.children[0] is None:
                return self
            return self.children[0].sift(s, teacher)
        else:
            if self.children[1] is None:
                return self
            return self.children[1].sift(s, teacher)
