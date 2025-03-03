import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

column = "equivalence_queries"

# Create a DataFrame for Seaborn
files: dict[str, str] = {
    "ttt": "ttt_matching_results.csv",
    "l_star": "l_star_results.csv",
}

dfs = []

for algorithm, filename in files.items():
    df = pd.read_csv(filename)
    df["algorithm"] = algorithm
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

# Create the Seaborn box plot
plt.figure(figsize=(8, 5))
sns.violinplot(x="algorithm", y=column, data=df)

# Labels
plt.title("Comparison of Algorithm Performance")
plt.ylabel(f"Unique {column}")

# Show the plot
# plt.show()
plt.savefig(f"{column}_box_plot.png")
