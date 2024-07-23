from observation_pack import run_alg as run_op
from l_star import run_alg as run_ls

# List of regex patterns
regex_patterns = [
    "a",
    "(ab|ba)*",
    "(aa)*(bb)*a",
    "a(bb)*bb",
    "a(bb)*bb|aba*|(aa)*",
    "((b(aa)*b)|((a|b(aa)*ab)((bb)|(ba(aa)*ab))*(a|ba(aa)*b)))*",
]

def run_and_average(func, regex, iterations=10):
    results = []
    for _ in range(iterations):
        result = func(regex)
        results.append(result)
    return sum(results) / len(results)

def main():
    iterations = 10
    averages_op = []
    averages_ls = []

    for regex in regex_patterns:
        avg1 = run_and_average(run_op, regex, iterations)
        avg2 = run_and_average(run_ls, regex, iterations)
        averages_op.append(avg1)
        averages_ls.append(avg2)

    print(regex_patterns)
    print(averages_op)
    print(averages_ls)
if __name__ == "__main__":
    main()
