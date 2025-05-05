from sklearn.preprocessing import StandardScaler

def load_and_preprocess(df):
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    X = df[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("✅ Preprocessing complete.")
    return X_scaled, df
