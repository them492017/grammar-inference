import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

columns = ["membership_queries", "equivalence_queries", "f1"]

files = {
    "ttt": "results/output/ttt.csv",
    "l_star": "results/output/l_star.csv",
}

df_ttt = pd.read_csv(files["ttt"])
df_l_star = pd.read_csv(files["l_star"])

# Merge on the "pattern" column
df = df_ttt.merge(df_l_star, on="pattern", suffixes=("_ttt", "_l_star"))

# Compute differences
for column in columns:
    df[column + "_diff"] = df[column + "_ttt"] - df[column + "_l_star"]

    # Violin / Boxplot
    plt.figure(figsize=(8, 5))

    sns.violinplot(y=df[column + "_diff"])

    plt.axhline(0, color="red", linestyle="--", alpha=0.8)  # Add zero reference line

    plt.title(f"Difference in {column} (TTT - L*)")
    plt.ylabel("Difference")
    plt.xlabel(column)

    plt.savefig(f"results/output/{column}_difference_violin_plot.png")
    plt.close()

    # Histogram
    plt.figure(figsize=(8, 5))

    sns.histplot(df[column + "_diff"], bins=50, kde=True, color="blue", alpha=0.5)
    plt.yscale("log")

    plt.title(f"Difference in {column} (TTT - L*)")
    plt.ylabel("Difference")
    plt.xlabel(column)

    plt.savefig(f"results/output/{column}_difference_histogram.png")
    plt.close()
