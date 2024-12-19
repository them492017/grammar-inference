from __future__ import annotations
from typing import Callable, TYPE_CHECKING
from math import floor, log

if TYPE_CHECKING:
    from state import Hypothesis
    from teach import Teacher

class Refiner:
    teacher: Teacher
    hypothesis: Hypothesis

    def __init__(self, teacher: Teacher, hypothesis: Hypothesis) -> None:
        self.teacher = teacher
        self.hypothesis = hypothesis

    def decompose(self, w: str, teacher: Teacher) -> tuple[str, str, str]:
        """
        Given a counterexample w, returns a decomposition (u, a, v) where
        len(a) == 1 and ...
        """
        # define prefix mapping
        def prefix_mapping(s: str, i: int) -> str:
            assert 0 <= i <= len(s)
            return self.hypothesis.run_non_deterministic(s[:i], w, teacher).aseq + s[i:]

        # define alpha
        def alpha(i: int) -> bool:
            return self.teacher.is_member(prefix_mapping(w, i)) == self.hypothesis.evaluate_non_deterministic(w, teacher)

        # binary search (or some variation of it)
        # i = self.exponential_search(alpha, len(w))
        i = self.binary_search(alpha, len(w))  # favours shorter suffixes
        print(i)

        return w[:i], w[i], w[i+1:]

    def binary_search(self, alpha: Callable[[int], bool], high: int, low: int = 0) -> int:
        while high - low > 1:
            mid = (low + high) // 2

            if alpha(mid) == 0:
                low = mid
            else:
                high = mid

        return low

    def exponential_search(self, alpha: Callable[[int], bool], high: int) -> int:
        range_len = 1
        low = 0
        found = False

        while not found and high - range_len > 0:
            print(low, high, high - range_len)
            if alpha(high - range_len) == 0:
                low = high - range_len
                found = True
            else:
                high -= range_len
                range_len *= 2

        return self.binary_search(alpha, high, low)

    def partition_search(self, alpha: Callable[[int], bool], max: int) -> int:
        step = floor(max / log(max, 2))
        low, high = 0, max
        found = False

        while not found and high - step > low:
            if alpha(high - step) == 0:
                low = high - step
                found = True
                break
            else:
                high = high - step

        return self.rs_eager_search(alpha, high, low)

    def rs_eager_search(self, alpha: Callable[[int], bool], high: int, low: int = 0) -> int:
        def beta(i: int) -> int:
            # TODO: could memoise previously computed alpha values
            return alpha(i) + alpha(i+1)

        while high > low:
            mid = (low + high) // 2

            if beta(mid) == 1:  # alpha(mid) != alpha(mid+1)
                return mid
            elif beta(mid) == 0:  # beta(mid+1) <= 1
                low = mid + 1
            else:  # beta(mid - 1) >= 1
                high = mid - 1

        return low
