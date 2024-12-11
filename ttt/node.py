from __future__ import annotations
from typing import Literal, Optional, Protocol

from tt import Teacher
from state import State
from node import InnerNode


class NodeProtocol(Protocol):
    """
    A node in the discrimination tree
    """
    children: tuple[Optional[NodeProtocol], Optional[NodeProtocol]]
    parent_info = Optional[tuple[InnerNode, bool]]
    
    def __init__(self, children: tuple[Optional[NodeProtocol], Optional[NodeProtocol]]) -> None:
        self.children = children
        self.parent_info = None

    @classmethod
    def make_leaf(cls) -> LeafNode:
        return LeafNode((None, None))

    @classmethod
    def make_inner(
        cls,
        discriminator: str,
        children: tuple[Optional[NodeProtocol], Optional[NodeProtocol]] = (None, None),
    ) -> InnerNode:
        return InnerNode(children, discriminator)

    @property
    def signature(self) -> list[tuple[str, bool]]:  # bool vs int?
        if self.parent_info is None:
            return []
        else:
            parent, child_subtree = self.parent_info  # TODO: type inference why?

            return [(parent.discriminator, child_subtree), *parent.signature]

    def sift(self, s: str, teacher: Teacher) -> NodeProtocol:
        ...


class Node:
    node_type: Literal['Leaf', 'Inner']
    leaf: Optional[LeafNode]
    inner: Optional[InnerNode]

    def __init__(
        self,
        node_type: Literal['Leaf', 'Inner'],
        leaf: Optional[LeafNode],
        inner: Optional[InnerNode]
    ) -> None:
        self.node_type = node_type
        self.leaf = leaf
        self.inner = inner

        match self.node_type:
            case 'Leaf':
                assert self.leaf is not None
            case 'Inner':
                assert self.inner is not None

    def split_leaf(self, discriminator: str) -> tuple[Node, Node]:
        if self.node_type != 'Leaf':
            raise ValueError("Can only call split_leaf on a LeafNode")




class LeafNode(NodeProtocol):
    state: Optional[State]

    def __init__(self, children: tuple[Optional[NodeProtocol], Optional[NodeProtocol]]) -> None:
        super().__init__(children)
        self.state = None

    def sift(self, s: str, teacher: Teacher) -> NodeProtocol:
        return self


class InnerNode(NodeProtocol):
    children: tuple[Optional[NodeProtocol], Optional[NodeProtocol]]
    discriminator: str

    def __init__(self, children: tuple[Optional[NodeProtocol], Optional[NodeProtocol]],
                 discriminator: str) -> None:
        super().__init__(children)
        self.discriminator = discriminator

    def sift(self, s: str, teacher: Teacher) -> NodeProtocol:
        if teacher.is_member(s):
            if self.children[0] is None:
                return self
            return self.children[0].sift(s, teacher)
        else:
            if self.children[1] is None:
                return self
            return self.children[1].sift(s, teacher)
