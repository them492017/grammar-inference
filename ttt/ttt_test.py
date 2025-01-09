from regex_enumerator.generate import generate
from regex_parser.regex import Regex
from ttt.teach import PerfectTeacher
from ttt.ttt import TTTAlgorithm

if __name__ == "__main__":
    alphabet = "ab"
    file = open("ttt_test.log", "w")

    try:
        for pattern in generate(alphabet):
            teacher = PerfectTeacher(alphabet, pattern)

            print(f"Learning [{pattern}] over alphabet [{alphabet}]")
            try:
                ttt = TTTAlgorithm(teacher, alphabet)
                hypothesis, dtree = ttt.learn()
            except AssertionError as e:
                print("=" * 50)
                print("Could not learn language")
                print(pattern)
                teacher.dfa.visualize('teacher_dfa')
                raise e

            dfa = hypothesis.to_dfa()

            regex = Regex.parse(pattern)
            regex.to_nfa().visualize(filename="regex_nfa")
            regex_dfa = regex.to_nfa().determinise()

            print(f"Pattern: {pattern}, Equivalent: {dfa.is_equivalent(regex_dfa)}", file=file)
    except KeyboardInterrupt:
        file.close()
