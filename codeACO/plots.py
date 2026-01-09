import matplotlib.pyplot as plt 
import numpy as np




results_human_prompt = "../results/statsHumanPrompt.csv"
results_aco_prompt = "../results/statsACO.csv"




import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_results(results: str, levels_names=""):
    # Load CSV
    df = pd.read_csv(results)

    # Optional filter by level_names
    if levels_names:
        df = df[df["level_names"] == levels_names]

    if df.empty:
        raise ValueError("No rows found for the given level_names")

    # Metrics to plot
    metrics = ["faithfulness", "relevance", "rouge"]

    # Extract mean value from "mean ± std"
    for m in metrics:
        df[m] = (
            df[m]
            .astype(str)
            .str.split("±")
            .str[0]
            .astype(float)
        )

    # Title: prompt with human == True
    if not (df["human"] == True).any():
        raise ValueError("No human prompt found for the selected level_names")

    title_prompt = df[df["human"] == True]["prompt"].iloc[0]

    # Strategy handling
    df["strategy"] = df["strategy"].fillna("Human")

    strategies = df["strategy"].unique()
    n_strategies = len(strategies)
    n_metrics = len(metrics)

    x = np.arange(n_metrics)
    width = 0.8 / n_strategies

    plt.figure(figsize=(10, 6))

    # Grouped bar chart
    for i, strategy in enumerate(strategies):
        subset = df[df["strategy"] == strategy]
        values = [subset[m].mean() for m in metrics]

        plt.bar(
            x + i * width,
            values,
            width,
            label=strategy
        )

    # Axes & labels
    plt.xticks(x + width * (n_strategies - 1) / 2, metrics)
    plt.xlabel("Metrics")
    plt.ylabel("Score")
    plt.title(title_prompt)

    # Legend
    plt.legend(title="Strategy")

    plt.tight_layout()
    plt.show()

plot_results(
    results="../results/results.csv",
    levels_names="TaskInstruction Input"
)