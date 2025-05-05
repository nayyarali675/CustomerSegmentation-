import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("🛍️ Mall Customer Segmentation")

import os
df = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/Mall_Customers.csv'))


if st.checkbox("Show Raw Data"):
    st.dataframe(df)

st.subheader("Age Distribution")
sns.histplot(df['Age'], kde=True)
st.pyplot()

st.subheader("Income vs Spending Score")
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=df)
st.pyplot()
