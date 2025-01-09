"""
S → Q | A | T | C | K
Q → A + ε | T + ε | C + ε
A → T + AT | C + AC | K + AK
AT → T | T + AT | AC
AC → C | C + AC | AK
AK → K | K + AK
T → a1 | a2 | · · · | ak
C → C0 C0 | C0 C
C0 → (Q) | (A) | T | K
K → (A) ∗ | T ∗ | (C) ∗
"""
EPSILON = "ε"

def make_cfg(alphabet: str) -> dict[str, list[str]]:
    return {
        "<S>": ["<Q>", "<A>", "<T>", "<C>", "<K>"],
        "<Q>": [f"(<A>|{EPSILON})", f"(<T>|{EPSILON})", f"(<C>|{EPSILON})"],
        "<A>": ["(<T>|<AT>)", "(<C>|<AC>)", "<K>|<AK>"],
        "<AT>": ["<T>", "(<T>|<AT>)", "<AC>"],
        "<AC>": ["<C>", "(<C>|<AC>)", "<AK>"],
        "<AK>": ["<K>", "<K>|<AK>"],
        "<T>": list(alphabet),
        "<C>": ["(<C0><C0>)", "(<C0><C>)"],
        "<C0>": ["(<Q>)", "(<A>)", "<T>", "<K>"],
        "<K>": ["((<A>)*)", "((<T>)*)", "((<C>)*)"],
    }
