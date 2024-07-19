class Node:
    def __init__(self, discriminator = None, zero = None, one = None):
        # leaf or inner
        self.children = []
        if zero is not None or one is not None:
            self.children[0] = zero
            self.children[1] = one
        if zero is not None:
            zero.parent = self
        if one is not None:
            one.parent = self

        self.parent = None

        self.state = None
        self.discriminator = None

        self.incoming = []

class DiscriminationTree:
    def __init__(self):
        self.tree = Node()

    def sift(self, u: str): ...
