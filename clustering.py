from sklearn.cluster import KMeans

def perform_kmeans(X_scaled, df, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    print(f"✅ KMeans clustering done with {n_clusters} clusters.")
    return df, kmeans
