from __future__ import annotations
import sys


ALPHABET = "ab"


class Regex:
    name: str
    regex: list

    def __str__(self) -> str:
        return f"{self.name}({", ".join(str(self.regex))})"

class Empty(Regex):
    def __init__(self) -> None:
        self.regex = []
        self.name = "Empty"

    def __str__(self) -> str:
        return "Empty()"

class Epsilon(Regex):
    def __init__(self) -> None:
        self.regex = []
        self.name = "Epsilon"

    def __str__(self) -> str:
        return "Epsilon()"

class Char(Regex):
    def __init__(self, char: str) -> None:
        assert len(char) == 1
        assert char in ALPHABET
        self.char = char

    def __str__(self) -> str:
        return f"Char({self.char})"

class Or(Regex):
    def __init__(self, r1: Regex, r2: Regex) -> None:
        self.r1 = r1
        self.r2 = r2
        self.regex = [self.r1, self.r2]
        self.name = "Union"

    def __str__(self) -> str:
        return f"Union({self.r1, self.r2})"

class And(Regex):
    def __init__(self, r1: Regex, r2: Regex) -> None:
        self.r1 = r1
        self.r2 = r2
        self.regex = [self.r1, self.r2]
        self.name = "Concat"

    def __str__(self) -> str:
        return f"Concast({self.r1, self.r2})"

class Star(Regex):
    def __init__(self, r: Regex) -> None:
        self.r = r

    def __str__(self) -> str:
        return f"Star({self.r})"


def parse(s: str) -> Regex:
    # Assume parentheses are everywhere
    print(f"parsing '{s}'")

    if len(s) == 0:
        return Epsilon()
    if len(s) == 1:
        return Char(s)

    parenthesis_positions = [0, 0, 0, 0]
    curr_idx = 0
    parenthesis_level = 0
    for i, c in enumerate(s):
        if c == "(":
            if parenthesis_level == 0:
                parenthesis_positions[curr_idx] = i
                curr_idx += 1

            parenthesis_level += 1
        if c == ")":
            parenthesis_level -= 1
        if c == "*" and curr_idx == 2:
            r = parse(s[parenthesis_positions[0]:parenthesis_positions[1]])
            return Star(r)
        
        if parenthesis_level == 0:
            parenthesis_positions[curr_idx] = i
            curr_idx += 1

        print(i, c, curr_idx, parenthesis_level)

        if curr_idx >= 4:
            break

    if curr_idx == 4:
        r1 = parse(
            s[(parenthesis_positions[0]+1):parenthesis_positions[1]]
        )
        r2 = parse(
            s[(parenthesis_positions[2]+1):parenthesis_positions[3]]
        )
        if parenthesis_positions[1] + 1 == parenthesis_positions[2]:
            return And(r1, r2)
        elif parenthesis_positions[1] + 2 == parenthesis_positions[2] and s[parenthesis_positions[1] + 1] == "|":
            return Or(r1, r2)
        else:
            dump = f"""
                {s}
                {parenthesis_positions}
                {curr_idx}
                {parenthesis_level}
            """
            raise ValueError(f"Could not parse {s}: Invalid separator\n{dump}")
    elif curr_idx == 2:
        return parse(
            s[(parenthesis_positions[0]+1):parenthesis_positions[1]]
        )
    else:
        dump = f"""
            {s}
            {parenthesis_positions}
            {curr_idx}
            {parenthesis_level}
        """
        raise ValueError(f"Could not parse {s}:\n{dump}")


def parse2(s: str) -> Regex:
    if len(s) == 0:
        return Epsilon()
    if len(s) == 1:
        return Char(s)

    curr_idx = 0
    symbols = []

    while curr_idx < len(s):
        if s[curr_idx] == "(":
            end_idx = matching_idx(s, curr_idx)
            symbols.append(parse2(s[(curr_idx+1) : (end_idx-1)]))
            curr_idx = end_idx + 1
        else:
            end_idx = s[curr_idx:].find("(")
            symbols.append(s[curr_idx:end_idx])
            curr_idx = end_idx

    # join symbols list
    while len(symbols) > 1:
        print(symbols)
        new_symbols = []
        i = 0

        while i < len(symbols) - 2:
            curr_symbol = symbols[i]
            next_symbol = symbols[i+1]
            next_next_symbol = symbols[i+2]

            if isinstance(curr_symbol, Regex) and isinstance(next_symbol, Regex):
                new_symbols.append(And(curr_symbol, next_symbol))
                i += 2
            elif isinstance(curr_symbol, Regex) and next_symbol == "|":
                new_symbols.append(Or(curr_symbol, next_next_symbol))
                i += 3
            elif isinstance(curr_symbol, Regex) and next_symbol == "*":
                new_symbols.append(Star(curr_symbol))
                i += 2
            elif isinstance(curr_symbol, str) and len(curr_symbol) == 1:
                new_symbols.append(Char(curr_symbol))
                i += 1
            elif isinstance(curr_symbol, str) and len(curr_symbol) == 0:
                new_symbols.append(Epsilon())
                i += 1
            else:
                raise ValueError("Could not parse string")

        symbols = new_symbols

    if len(symbols) == 0:
        raise ValueError("Could not parse string")
    else:
        return symbols[0]


def matching_idx(s: str, i: int) -> int:
    print("Matching Idx", s, i)
    if s[i] != "(":
        raise ValueError("Initial character is not '('")

    depth = 1

    for j, c in enumerate(s[i+1:], start=i+1):
        if c == "(":
            depth += 1
        if c == ")":
            depth -= 1
        if depth == 0:
            return j

    raise ValueError("Not match found")


if __name__ == "__main__":
    regex = parse2(sys.argv[1])

    print("The regex was")
    print(regex)
