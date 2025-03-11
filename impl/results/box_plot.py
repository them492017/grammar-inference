import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# pattern,unique_membership_queries,membership_queries,equivalence_queries,precision,recall,f1
columns = ["membership_queries", "equivalence_queries", "f1"]

files: dict[str, str] = {
    "ttt": "results/output/ttt.csv",
    "l_star": "results/output/l_star.csv",
}

dfs = []

for algorithm, filename in files.items():
    df = pd.read_csv(filename)
    df["algorithm"] = algorithm
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

for column in columns:
    plt.figure(figsize=(8, 5))
    sns.violinplot(x="algorithm", y=column, data=df)

    plt.title("Comparison of Algorithm Performance")
    plt.ylabel(f"Unique {column}")

    plt.savefig(f"results/output/{column}_box_plot.png")
