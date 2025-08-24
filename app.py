
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


# Function: Load and preprocess dataset

@st.cache_data
def load_data():
    df = pd.read_csv("spotify dataset.csv")
    df.dropna(subset=['track_name', 'track_artist'], inplace=True)
    df['track_album_release_date'] = pd.to_datetime(df['track_album_release_date'], errors='coerce')
    df['track_album_release_date'] = df['track_album_release_date'].fillna(df['track_album_release_date'].median())
    return df


# Nearest Neighbors model SetUp

def setup_model(df):
    features = ['danceability', 'energy', 'speechiness', 'acousticness',
                'instrumentalness', 'liveness', 'valence', 'tempo']#important audio features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    #fit-learns mean and standard deviation of the data
    #transform-applies that scaling
    #fit.transform is the shortcut that does both at the same time
    model = NearestNeighbors(n_neighbors=6, algorithm='auto')
    #auto-tells scikit-learn to automatically choose the best algorithm based on the dataset
    model.fit(X_scaled)
    return model, X_scaled, features


# Function: Recommend Songs

def recommend_songs(df, model, X_scaled, features, track_name, artist_name):
    idx = df[(df['track_name'].str.lower() == track_name.lower()) &
             (df['track_artist'].str.lower() == artist_name.lower())].index
    if len(idx) == 0:
        return None
    distances, indices = model.kneighbors([X_scaled[idx[0]]])
    return df.iloc[indices[0][1:]][['track_name', 'track_artist', 'playlist_genre']]


# Streamlit App UI

st.title("🎧 Spotify Music Recommender System")

df = load_data()
model, X_scaled, features = setup_model(df)

track_input = st.text_input("Enter a track name:")
artist_input = st.text_input("Enter the artist name:")

if st.button("Get Recommendations"):
    if not track_input or not artist_input:
        st.warning("Please enter both track name and artist name.")
    else:
        results = recommend_songs(df, model, X_scaled, features, track_input, artist_input)
        if results is not None:
            st.subheader("Recommended Songs:")
            st.dataframe(results)
        else:
            st.error("Track not found. Please check spelling or try another song.")
