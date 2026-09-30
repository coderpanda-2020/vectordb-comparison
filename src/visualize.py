import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_visualizations():
    df = pd.read_csv("results/benchmark_results.csv")
    
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Vector Database Benchmark Comparison', fontsize=16)
    
    # 1. Indexing Time
    sns.barplot(data=df, x="Volume", y="Index Time (s)", hue="DB", ax=axes[0, 0])
    axes[0, 0].set_title('Indexing Time by Volume')
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_ylabel('Time (seconds) - Log Scale')
    
    # 2. Query Latency
    sns.barplot(data=df, x="Volume", y="Query Latency (s)", hue="DB", ax=axes[0, 1])
    axes[0, 1].set_title('Query Latency by Volume')
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_ylabel('Latency (seconds) - Log Scale')
    
    # 3. Memory Usage
    sns.barplot(data=df, x="Volume", y="Memory (MB)", hue="DB", ax=axes[1, 0])
    axes[1, 0].set_title('Memory Usage by Volume')
    axes[1, 0].set_ylabel('Memory (MB)')
    
    # 4. Recall
    sns.barplot(data=df, x="Volume", y="Recall", hue="DB", ax=axes[1, 1])
    axes[1, 1].set_title('Recall by Volume')
    axes[1, 1].set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('results/benchmark_visualizations.png', dpi=300)
    print("Visualizations saved to results/benchmark_visualizations.png")

if __name__ == "__main__":
    create_visualizations()
