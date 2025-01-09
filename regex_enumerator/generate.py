from typing import Deque, Generator

from regex_enumerator.regex_cfg import make_cfg
from regex_parser.regex import Regex


def generate(alphabet: str) -> Generator[str, None, None]:
    cfg = make_cfg(alphabet)
    queue: Deque[list[str]] = Deque()
    queue.append(["<S>"])

    while True:
        curr = queue.popleft()

        for i, symbol in enumerate(curr):
            if len(symbol) > 1:  # symbol is a variable
                queue.extend(
                    [
                        curr[:i] + production + curr[i + 1 :]
                        for production in cfg[symbol]
                    ]
                )
                break
        else:
            yield "".join(curr)


if __name__ == "__main__":
    for s in generate("ab"):
        print(Regex.parse(s))
