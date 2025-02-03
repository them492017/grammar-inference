from __future__ import annotations

from typing import TYPE_CHECKING, Generator, Optional

from ttt.print import print


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
    _block: Optional[Node]
    _incoming_non_tree: set[Transition]
    depth: int

    def __init__(self,
        is_leaf: bool,
        children: tuple[Optional[Node], Optional[Node]],
        parent: Optional[Node] = None,
        state: Optional[State] = None,
        discriminator: Optional[str] = None,
    ) -> None:
        self.is_leaf = is_leaf
        self._is_temporary = discriminator != "" and not self.is_leaf
        print("new node is temporary:", self._is_temporary)
        self._children = children
        self._parent = parent
        self._state = state
        self._discriminator = discriminator
        self._block = None
        self._incoming_non_tree = set()
        self.depth = 0

    def replace_with_final(self, node: Node) -> None:
        # replace the node with another in place
        print(f"Replacing {self} with {node}")
        assert self.is_leaf == node.is_leaf

        self.is_leaf = node.is_leaf
        self._is_temporary = False
        self.block = None

        self._children = node._children
        self._state = node._state
        self._discriminator = node._discriminator
        self._incoming_non_tree = node._incoming_non_tree

    # def __repr__(self) -> str:
    #     base = f"Node<d={"~" if self._is_temporary else ""}'{self._discriminator}' state={self._state}"
    #     if self.block and not self.block.is_leaf:
    #         return base + f" block={self.block._discriminator}>"
    #     elif self.block:
    #         return base + f" block={self.block._state}>"
    #     else:
    #         return base + ">"

    def __repr__(self) -> str:
        if self.is_leaf:
            return f"Node<{self.state}>"
        elif self.is_temporary:
            return f"Node<~{self.discriminator}>"
        else:
            return f"Node<{self.discriminator}>"

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
        min_depth = min(map(lambda node: node.depth, nodes))
        nodes_in_layer: set[Node] = set()

        for node in nodes:
            print(node, node.depth)
            while node.depth > min_depth:
                node = node.parent

                if node is None:
                    raise ValueError("Null parent of non-root node")

            nodes_in_layer.add(node)

        while len(nodes_in_layer) > 1:
            nodes_in_layer = {
                node.parent for node in nodes_in_layer if node.parent is not None
            }

        if len(nodes_in_layer) == 0:
            raise ValueError(f"LCA of {nodes} couldn't be computed")

        print(f"lca of {nodes} is {list(nodes_in_layer)[0]}")
        return nodes_in_layer.pop()


    def print_tree(self, child: int = -1, level: int = 0):
        """
        A method that outputs a discrimination tree rooted at `self`. From
        https://stackoverflow.com/questions/34012886/print-binary-tree-level-by-level-in-python
        """
        if child == 1:
            arrow = "=>"
        elif child == 0:
            arrow = "->"
        else:
            arrow = "->"

        if self.is_leaf:
            print(f"{" " * 4 * level}{arrow} {self}")
        else:
            assert self.children[0]
            assert self.children[1]
            self.children[0].print_tree(0, level+1)
            print(f"{" " * 4 * level}{arrow} {self}")
            self.children[1].print_tree(1, level+1)

    def __iter__(self) -> Generator[Node, None, None]:
        if self._children[0]:
            for node in self._children[0]:
                yield node
        yield self
        if self._children[1]:
            for node in self._children[1]:
                yield node

    def states(self) -> Generator[State, None, None]:
        for node in self:
            if node.is_leaf and node.state is not None:
                yield node.state

    @property
    def is_temporary(self) -> bool:
        assert not self.is_leaf
        return self._is_temporary

    @property
    def parent(self) -> Optional[Node]:
        return self._parent

    @parent.setter
    def parent(self, node: Node) -> None:
        print(f"Setting parent of {self}")
        print(self.parent)
        self._parent = node
        self.block = node.block
        self.depth = node.depth + 1
        print(self.parent)

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
        print(f"Changing state of {self} from {self.state} to {state}")
        assert self.is_leaf
        self._state = state
        if self.block is None:
            self.block = self

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

    @property
    def block(self) -> Optional[Node]:
        return self._block

    @block.setter
    def block(self, block: Optional[Node]) -> None:
        print(f"Changing block of {self} from {self.block} to {block}")
        self._block = block

    @property
    def incoming_non_tree(self) -> set[Transition]:
        return self._incoming_non_tree

    @incoming_non_tree.setter
    def incoming_non_tree(self, inc: set[Transition]) -> set[Transition]:
        for t in list(inc):
            t.target_node = self
        return self._incoming_non_tree

    def link(self, state: State) -> None:
        self.state = state
        # if state.node:
        #     self.incoming_non_tree |= state.node.incoming_non_tree
        #     self.incoming_tree |= state.node.incoming_tree
        state.node = self

    def split_leaf(self, discriminator: str) -> tuple[Node, Node]:
        assert self.is_leaf
        self.is_leaf = False
        self._is_temporary = True
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
        if self.is_leaf or self.is_temporary:
            return self

        subtree = int(teacher.is_member(s + self.discriminator))
        child = self.children[subtree]

        assert child is not None

        return child.soft_sift(s, teacher)
