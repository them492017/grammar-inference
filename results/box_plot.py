import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create a DataFrame for Seaborn
files: dict[str, str] = {
    "ttt": "ttt_test.csv",
    "observation_pack": "observation_pack_test.csv",
}

dfs = []

for algorithm, filename in files.items():
    df = pd.read_csv(filename)
    df["algorithm"] = algorithm
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

# Create the Seaborn box plot
plt.figure(figsize=(8, 5))
sns.violinplot(x="algorithm", y="unique_membership_queries", data=df)

# Labels
plt.title("Comparison of Algorithm Performance")
plt.ylabel("Unique membership queries")

# Show the plot
# plt.show()
plt.savefig("membership_query_box_plot.png")
