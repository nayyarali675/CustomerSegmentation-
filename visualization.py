import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd

def plot_clusters(X_scaled, df):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['PCA1'], df['PCA2'] = X_pca[:, 0], X_pca[:, 1]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='tab10', s=100)
    plt.title('Customer Segments')
    plt.savefig('output/cluster_plot.png')
    plt.close()
    print("✅ Cluster plot saved to output/cluster_plot.png")

def describe_clusters(df):
    print("📊 Cluster Summary:")
    summary = df.groupby('Cluster').mean(numeric_only=True)[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    print(summary)
