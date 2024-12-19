import unittest
from node import Node


class TestNodeMethods(unittest.TestCase):
    num_nodes: int

    def __init__(self):
        super().__init__()
        self.num_nodes = 0

    def node_builder(self, leaf: bool = False) -> Node:
        self.num_nodes += 1
        if leaf:
            return Node.make_leaf()
        else:
            return Node.make_inner("a" * self.num_nodes)

    def tree_builder(self, tree_str: str) -> Node:
        """
        Converts a string input of a tree into a discrimination tree

        e.g. ...,,,,.,,.,, becomes
                    N
                   / \
                  N   N
                 / \
                N   N
        """
        ...

    def test_lca_empty(self):
        with self.assertRaises(AssertionError):
            Node.lca([])

    def test_lca_trivial(self):
        Node.lca([])


if __name__ == "__main__":
    unittest.main()
