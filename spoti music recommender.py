import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

# Load dataset
df = pd.read_csv("spotify dataset.csv")  # Replace with your actual CSV file path

# Step 1: Data Preprocessing
df.dropna(subset=['track_name', 'track_artist'], inplace=True)
df['track_album_release_date'] = pd.to_datetime(df['track_album_release_date'], errors='coerce')
df['track_album_release_date'] = df['track_album_release_date'].fillna(df['track_album_release_date'].median())
df.reset_index(drop=True, inplace=True)

# Step 2: Data Analysis and Visualization
plt.figure(figsize=(10, 6))
sns.histplot(df['track_popularity'], bins=30, kde=True)
plt.title('Track Popularity Distribution')
plt.xlabel('Popularity')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Step 3: Correlation Matrix
numerical_features = df.select_dtypes(include=['float64', 'int64']).drop(columns=['key', 'mode'])
corr = numerical_features.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# Step 4: Clustering using KMeans
features = ['danceability', 'energy', 'speechiness', 'acousticness',
            'instrumentalness', 'liveness', 'valence', 'tempo']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

sil_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    sil_scores.append(silhouette_score(X_scaled, labels))

optimal_k = np.argmax(sil_scores) + 2
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df['pca1'] = pca_result[:, 0]
df['pca2'] = pca_result[:, 1]

plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='pca1', y='pca2', hue='cluster', palette='tab10')
plt.title("Clusters of Songs (PCA projection)")
plt.tight_layout()
plt.show()

# Step 5: Recommendation System using Nearest Neighbors
nn_model = NearestNeighbors(n_neighbors=6, algorithm='auto')
nn_model.fit(X_scaled)

def recommend_songs(track_name, artist_name):
    idx = df[(df['track_name'].str.lower() == track_name.lower()) &
             (df['track_artist'].str.lower() == artist_name.lower())].index
    if len(idx) == 0:
        return f"Track '{track_name}' by '{artist_name}' not found."
    
    distances, indices = nn_model.kneighbors([X_scaled[idx[0]]])
    recommendations = df.iloc[indices[0][1:]][['track_name', 'track_artist', 'playlist_genre']]
    return recommendations

# Example usage:
recommendations = recommend_songs("bad guy (with Justin Bieber)", "Billie")
print(recommendations)
