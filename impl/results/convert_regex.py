EPSILON = "\u03b5"


def convert_epsilon_regex(pattern: str) -> str:
    return pattern.replace(EPSILON, "(?:)")