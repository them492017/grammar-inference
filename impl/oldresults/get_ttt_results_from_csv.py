import csv
from ttt.ttt_test import learn_pattern_with_stats_without_dfa_check

csv_filename = "results/l_star_results.csv"
alphabet = "ab"

with open(csv_filename, mode="r") as file, open("ttt_matching_results.csv", mode="w") as output_file:
    reader = csv.reader(file)
    next(reader)

    print("pattern,unique_membership_queries,membership_queries,equivalence_queries,success", file=output_file)

    # Loop through each row and get the first column (pattern)
    try:
        for row in reader:
            pattern = row[0]
            learn_pattern_with_stats_without_dfa_check(pattern, output_file)
    except KeyboardInterrupt:
        pass
