import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Streamlit App Title
st.title("Customer Segmentation using PCA and K-Means")

# Load dataset directly
file_path = "wine.csv"  # Ensure the file is in the same directory
df = pd.read_csv(file_path)

# Display dataset preview
st.write("### Dataset Preview", df.head())

# Extract feature columns (excluding Customer_Segment if present)
if "Customer_Segment" in df.columns:
    df = df.drop(columns=["Customer_Segment"])

# Select numeric columns for PCA
features = df.select_dtypes(include=[np.number])

# Standardize data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

# User input for PCA components
max_components = min(10, scaled_data.shape[1])
n_components = st.slider("Select Number of Principal Components:", 2, max_components, 2)

# Apply PCA
pca = PCA(n_components=n_components)
pca_result = pca.fit_transform(scaled_data)

# Explained variance plot
explained_variance = pca.explained_variance_ratio_
st.write("### Explained Variance by Principal Components")
fig, ax = plt.subplots()
ax.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7)
ax.set_xlabel("Principal Component")
ax.set_ylabel("Variance Explained")
ax.set_title("PCA Explained Variance")
st.pyplot(fig)

# Convert PCA result into DataFrame
pca_df = pd.DataFrame(pca_result, columns=[f"PC{i+1}" for i in range(n_components)])

# Clustering using K-Means
n_clusters = st.slider("Select Number of Clusters for Segmentation:", 2, 10, 3)
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
pca_df["Cluster"] = kmeans.fit_predict(pca_result)

# Scatter plot of first two principal components
st.write("### Customer Segmentation (First Two Principal Components)")
fig, ax = plt.subplots()
sns.scatterplot(x=pca_df["PC1"], y=pca_df["PC2"], hue=pca_df["Cluster"], palette="viridis", ax=ax)
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_title("Customer Segments")
st.pyplot(fig)

# Show transformed data with clusters
st.write("### PCA Transformed Data with Clusters")
st.write(pca_df.head())

# Download PCA results with clusters
csv = pca_df.to_csv(index=False).encode("utf-8")
st.download_button("Download Segmented Data", csv, "segmented_customers.csv", "text/csv")

# --- User Input for Prediction ---
st.write("### Predict Customer Segment for New Data")

# User input fields for all features
input_data = []
for col in features.columns:
    value = st.number_input(f"Enter {col}:", value=float(df[col].mean()))
    input_data.append(value)

# Standardize input data
input_data = np.array(input_data).reshape(1, -1)
scaled_input = scaler.transform(input_data)

# Apply PCA to input data
pca_input = pca.transform(scaled_input)

# Predict customer segment
predicted_cluster = kmeans.predict(pca_input)[0]
st.write(f"### Predicted Customer Segment: {predicted_cluster}")
