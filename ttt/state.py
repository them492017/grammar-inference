from transition import Transition


class State:
    """
    A state in the spanning-tree hypothesis
    """
    transitions: dict[str, Transition]
    # incoming_transition: TreeTransition

