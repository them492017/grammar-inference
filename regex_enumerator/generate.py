from typing import Deque, Generator

from regex_enumerator.regex_cfg import make_cfg


def generate(alphabet: str) -> Generator[str, None, None]:
    cfg = make_cfg(alphabet)
    queue: Deque[list[str]] = Deque()
    queue.append(["<S>"])
