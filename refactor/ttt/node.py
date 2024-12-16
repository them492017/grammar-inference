from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional, Any, cast

if TYPE_CHECKING:
    from state import State
    from teach import Teacher
    from transition import Transition


class Node:
    is_leaf: bool
    _is_temporary: bool
    _children: tuple[Optional[Node], Optional[Node]]
    _parent: Optional[Node]
    _state: Optional[State]
    _discriminator: Optional[str]
    incoming_non_tree: set[Transition]  # TODO: make this a list unless open_Transitions list is changed

    def __init__(self,
        is_leaf: bool,
        children: tuple[Optional[Node], Optional[Node]],
        parent: Optional[Node] = None,
        state: Optional[State] = None,
        discriminator: Optional[str] = None,
    ) -> None:
        self.is_leaf = is_leaf
        self._is_temporary = discriminator == ""
        self._children = children
        self._parent = parent
        self._state = state
        self._discriminator = discriminator
        self.incoming_non_tree = set()
        self.depth = 0

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

    @classmethod
    def lca(cls, nodes: list[Node]) -> Node:
        assert len(nodes) > 1

        seen = set()
        unique_nodes: list[Node] = []

        min_depth = min(map(lambda node: node.depth, nodes))
        for node in nodes:
            while node.depth > min_depth:
                seen.add(node)
                node = node.parent
                assert node is not None
                if node in seen:
                    break
            if node not in seen:
                unique_nodes.append(node)

        while len(unique_nodes) > 1:
            parent_nodes = list(set(map(lambda node: 
                node.parent, unique_nodes)))
            unique_parents = cast(list[Node], parent_nodes)  # parents cannot be None
            parent_nodes = unique_parents

        assert len(unique_nodes) == 1

        return unique_nodes[0]


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

    def states(self) -> Iterable[State]:
        return filter(lambda node: node.is_leaf, self.__iter__())

    @property
    def is_temporary(self) -> bool:
        assert not self.is_leaf
        return self._is_temporary

    @property
    def parent(self) -> Optional[Node]:
        return self._parent

    @parent.setter
    def parent(self, node: Node) -> None:
        self._parent = node
        self.depth = node.depth + 1

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
        child = self.children[subtree]
        assert child is not None

        return child.sift(s, teacher)

    def soft_sift(self, s: str, teacher: Teacher) -> Node:
        if self.is_leaf or not self.is_temporary:
            return self

        subtree = int(teacher.is_member(s + self.discriminator))
        child = self.children[subtree]

        assert child is not None

        # TODO: double chcek it should point to the block root
        return child.soft_sift(s, teacher)
