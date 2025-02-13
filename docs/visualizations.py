import matplotlib.pyplot as plt
import pandas as pd
import os

# Creating a pandas DataFrame
data = {
    "Benchmark/Models": ["ETT1", "ETT2", "Flu-US", "Flu-Japan", "PEM-Bays", "NY-Bike Demand", "NY-Taxi Demand", "Nasdaq", "M4"],
    "TimesFM": [0.58, 0.49, 1.32, 1214, 3.7, 2.8, 12.19, 0.22, 1.07],
    "Chronos": [0.59, 0.52, 1.21, 1228, 3.7, 3.1, 12.82, 0.27, 1.04],
    "MOIRAI": [0.62, 0.55, 1.31, 1336, 3.9, 3.5, 13.71, 0.24, 1.21],
    "Lag-LLAMA": [0.64, 0.57, 1.46, 1416, 3.9, 2.9, 13.43, 0.28, 1.33],
    "LPTM": [0.49, 0.46, 0.79, 704, 2.5, 2.4, 11.84, 0.17, 0.94]
}

df = pd.DataFrame(data)

# Create a folder to store the visualizations
output_folder = "benchmark_visualizations"
os.makedirs(output_folder, exist_ok=True)

# Generate and save bar graphs for each benchmark
def save_individual_benchmark_graphs():
    for benchmark in df["Benchmark/Models"]:
        subset = df[df["Benchmark/Models"] == benchmark].drop("Benchmark/Models", axis=1)
        plt.figure(figsize=(8, 5))
        bars = plt.bar(subset.columns, subset.values[0])
        
        # Emphasize LPTM with a different color
        for bar, label in zip(bars, subset.columns):
            if label == "LPTM":
                bar.set_color("green")
        
        plt.xlabel("Foundational Models")
        plt.ylabel("MAE Performance")
        plt.title(f"Model Performance on {benchmark} Benchmark Dataset")
        plt.xticks(rotation=45)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        save_path = os.path.join(output_folder, f"{benchmark}_performance.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

# Generate and save a combined bar graph for all benchmarks
def save_combined_benchmark_graph():
    ax = df.set_index("Benchmark/Models").plot(kind="bar", figsize=(12, 6), colormap="Set1")
    
    # Emphasize LPTM bars
    for container in ax.containers:
        for bar, label in zip(container, df.columns[1:]):
            if label == "LPTM":
                bar.set_color("red")
                bar.set_edgecolor("black")
                bar.set_linewidth(1.5)
    
    plt.xlabel("Benchmark Datasets", fontsize=12)
    plt.ylabel("Performance", fontsize=12)
    plt.title("Comparison of Model Performance Across Benchmarks", fontsize=14, fontweight="bold")
    plt.legend(title="Models", fontsize=10)
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    save_path = os.path.join(output_folder, "All_Benchmarks_Comparison.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

# Execute both functions
save_individual_benchmark_graphs()
save_combined_benchmark_graph()
