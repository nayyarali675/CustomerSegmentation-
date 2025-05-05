import pandas as pd

def run_eda():
    df = pd.read_csv('data/Mall_Customers.csv')
    df.columns = ['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    print("✅ Dataset Loaded. Shape:", df.shape)
    print(df.describe())
    print(df.info())
    return df
