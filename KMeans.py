#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os


# In[3]:


# Set OMP_NUM_THREADS to avoid memory leak warning
os.environ['OMP_NUM_THREADS'] = '1'

# Load prediction datasets from FF_Predictions folder
predictions_path = 'FF_Predictions'
prediction_files = {
    'K': os.path.join(predictions_path, 'K_predictions.csv'),
    'QB': os.path.join(predictions_path, 'QB_predictions.csv'),
    'RB': os.path.join(predictions_path, 'RB_predictions.csv'),
    'TE': os.path.join(predictions_path, 'TE_predictions.csv'),
    'WR': os.path.join(predictions_path, 'WR_predictions.csv')
}

# Load and combine prediction data
dfs = []
for position, file_path in prediction_files.items():
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['Position'] = position
        dfs.append(df)

combined_predictions = pd.concat(dfs, ignore_index=True)

# Select relevant features for clustering
features = [
    'Games_Played',
    'sample_weight',
    'predicted_points'
]

# Additional position-specific features
position_features = {
    'QB': ['Yds_avg', 'TD_avg', 'Int_avg', 'Rate_avg', 'FL_avg'],
    'RB': ['Yds_avg', 'TD_avg', 'Rec_avg', 'FL_avg'],
    'WR': ['Rec_avg', 'Yds_avg', 'TD_avg', 'FL_avg'],
    'TE': ['Rec_avg', 'Yds_avg', 'TD_avg', 'FL_avg'],
    'K': ['FG%_avg', 'XP%_avg', 'Pts_avg']
}

# Create separate clustering analyses for each position
for position in prediction_files.keys():
    print(f"\nProcessing {position}...")
    
    position_df = combined_predictions[combined_predictions['Position'] == position].copy()
    position_df = position_df.reset_index(drop=True)
    
    # Get relevant features for this position
    current_features = features + position_features.get(position, [])
    available_features = [f for f in current_features if f in position_df.columns]
    
    if len(position_df) < 2:
        print(f"Insufficient data for position {position}")
        continue
        
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(position_df[available_features])
    
    # Apply KMeans clustering
    n_clusters = min(len(position_df), 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    position_df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Print cluster analysis
    print(f"\nCluster Analysis for {position}:")
    for cluster in sorted(position_df['cluster'].unique()):
        cluster_data = position_df[position_df['cluster'] == cluster]
        print(f"\nCluster {cluster}:")
        print(f"Average predicted points: {cluster_data['predicted_points'].mean():.2f}")
        print(f"Range: {cluster_data['predicted_points'].min():.2f} - {cluster_data['predicted_points'].max():.2f}")
        print(f"Number of players: {len(cluster_data)}")
        print("Players:", ', '.join(sorted(cluster_data['Player'].tolist())))
        
        # Print average stats for the cluster
        print("\nCluster Averages:")
        for feature in available_features:
            print(f"{feature}: {cluster_data[feature].mean():.2f}")

print("\nClustering analysis complete.")


# In[ ]:




