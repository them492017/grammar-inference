from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Any

if TYPE_CHECKING:
    from state import State
    from teach import Teacher
    from transition import Transition


class Node:
    is_leaf: bool
    _children: tuple[Optional[Node], Optional[Node]]
    parent: Optional[Node]
    _state: Optional[State]
    _discriminator: Optional[str]
    incoming_non_tree: set[Transition]  # TODO: make this a list unless open_Transitions list is changed

    def __init__(self,
        is_leaf: bool,
        children: tuple[Optional[Node], Optional[Node]],
        parent: Optional[Node] = None,
        state: Optional[State] = None,
        discriminator: Optional[str] = None
    ) -> None:
        self.is_leaf = is_leaf
        self._children = children
        self.parent = parent
        self._state = state
        self._discriminator = discriminator
        self.incoming_non_tree = set()

    def __repr__(self) -> str:
        return f"Node<d='{self._discriminator}' state={self._state}>"

    @classmethod
    def make_leaf(cls) -> Node:
        return Node(True, (None, None))

    @classmethod
    def make_inner(
        cls,
        discriminator: str,
        children: tuple[Optional[Node], Optional[Node]] = (None, None),
    ) -> Node:
        new_node = Node(False, children, discriminator=discriminator)
        for child in children:
            if child:
                child.parent = new_node
        return new_node

    def print_tree(self, child: int = -1, level: int = 0, property: str = ""):
        """
        A method that outputs a discrimination tree rooted at `self`. Adapted from
        https://stackoverflow.com/questions/34012886/print-binary-tree-level-by-level-in-python
        """
        if child == 1:
            arrow = "=>"
        elif child == 0:
            arrow = "->"
        else:
            arrow = "->"

        if self.is_leaf:
            print(f"{" " * 4 * level}{arrow} <{self}>")
        else:
            assert self.children[0]
            assert self.children[1]
            self.children[0].print_tree(0, level+1, property)
            print(f"{" " * 4 * level}{arrow} <{self}>")
            self.children[1].print_tree(1, level+1, property)

    def __iter__(self) -> Any:
        if self._children[0]:
            for node in self._children[0]:
                yield node
        yield self
        if self._children[1]:
            for node in self._children[1]:
                yield node

    @property
    def children(self) -> tuple[Optional[Node], Optional[Node]]:
        assert not self.is_leaf
        return self._children

    @property
    def state(self) -> Optional[State]:
        assert self.is_leaf
        return self._state

    @state.setter
    def state(self, state: State) -> None:
        assert self.is_leaf
        self._state = state

    @property
    def discriminator(self):
        assert not self.is_leaf
        assert self._discriminator is not None
        return self._discriminator

    @property
    def parent_value(self) -> bool:
        if self.parent is None:
            raise ValueError(f"{self} has no parent")
        if self == self.parent.children[0]:
            return False
        if self == self.parent.children[1]:
            return True
        raise ValueError(f"{self} is not the child of its parent")

    @property
    def signature(self) -> list[tuple[str, bool]]:
        if self.parent is None:
            return []
        else:
            return [(self.parent.discriminator, self.parent_value), *self.parent.signature]

    def link(self, state: State) -> None:
        self.state = state
        state.node = self

    def split_leaf(self, discriminator: str) -> tuple[Node, Node]:
        assert self.is_leaf
        self.is_leaf = False
        self._state = None
        self._discriminator = discriminator
        children = (Node.make_leaf(), Node.make_leaf())
        for child in children:
            child.parent = self
        self._children = children
        return children

    def sift(self, s: str, teacher: Teacher) -> Node:
        if self.is_leaf:
            return self
        subtree = int(teacher.is_member(s + self.discriminator))
        assert self.children[subtree] is not None
        return self.children[subtree].sift(s, teacher)  # type: ignore
