import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
k_analysis = pd.read_csv('K_analysis.csv')
qb_analysis = pd.read_csv('QB_analysis.csv')
rb_analysis = pd.read_csv('RB_analysis.csv')
te_analysis = pd.read_csv('TE_analysis.csv')
wr_analysis = pd.read_csv('WR_analysis.csv')
games_updated = pd.read_csv('gamesUpdated.csv')

# Merge datasets on Player column
combined_data = pd.merge(k_analysis, qb_analysis, on='Player', how='outer', suffixes=('_k', '_qb'))
combined_data = pd.merge(combined_data, rb_analysis, on='Player', how='outer', suffixes=('', '_rb'))
combined_data = pd.merge(combined_data, te_analysis, on='Player', how='outer', suffixes=('_te', '_te'))
combined_data = pd.merge(combined_data, wr_analysis, on='Player', how='outer', suffixes=('_wr', '_wr'))

# Select relevant features for clustering
features = [
    'Yds_total', 'TD_total', 'Rec_total', 'Tgt_total', 'Att_total',
    'FGM_total', 'FGA_total', 'XP%_total', 'Games_Played'
]

# Ensure the features are present
features = [f for f in features if f in combined_data.columns]

# Debug missing values before handling
print("Missing values before handling:")
print(combined_data[features].isna().sum())
print(f"Rows before handling: {combined_data.shape[0]}")

# Fill missing values with column means
combined_data[features] = combined_data[features].fillna(combined_data[features].mean())

# Verify remaining rows
print(f"Rows after handling missing values: {combined_data.shape[0]}")

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(combined_data[features])

# Apply PCA to reduce dimensionality
pca = PCA(n_components=2)  # Reduce to 2 principal components for visualization
X_pca = pca.fit_transform(X_scaled)

# Check shape after PCA
print(f"Shape after PCA: {X_pca.shape}")

# Apply KMeans clustering on the PCA-transformed data
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(X_pca)

# Assign cluster labels to the DataFrame
combined_data['cluster'] = kmeans.labels_

# Print explained variance ratio
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# Analyze clusters by position
if 'Position' in combined_data.columns:
    position_cluster_comparison = pd.crosstab(combined_data['Position'], combined_data['cluster'])
    print("Position vs Cluster:")
    print(position_cluster_comparison)

# Visualize the clusters in 2D PCA space
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=combined_data['cluster'], cmap='viridis', s=50)
plt.title('KMeans Clusters in PCA-Reduced Space')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.show()

# Save the clustered data
combined_data.to_csv('clustered_players_with_pca.csv', index=False)
