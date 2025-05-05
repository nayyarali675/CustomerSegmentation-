from src.eda import run_eda
from src.preprocess import load_and_preprocess
from src.clustering import perform_kmeans
from src.visualization import plot_clusters, describe_clusters

def main():
    print("🚀 Running Customer Segmentation Pipeline...\n")
    df = run_eda()
    X_scaled, df = load_and_preprocess(df)
    df, kmeans_model = perform_kmeans(X_scaled, df)
    plot_clusters(X_scaled, df)
    describe_clusters(df)

if __name__ == "__main__":
    main()
