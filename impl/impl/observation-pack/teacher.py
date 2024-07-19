from hypothesis import Hypothesis

class Teacher:
    def __init__(self):
        ...

    def equivalent(self, h: Hypothesis) -> tuple[bool, str]: 
        ...

    def membership(self, u: str) -> bool: 
        ...
