import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from numpy import dot
from numpy.linalg import norm

def load_data():
    df = pd.read_csv("TV Shows - Netflix.csv")
    return df

def preprocess_data(df):
    X = df.drop('Titles', axis=1)
    le = LabelEncoder()
    X['IMDB_Rating'] = le.fit_transform(X['IMDB_Rating'])
    numerical_features = X.select_dtypes(include=np.number).columns
    X_numerical = X[numerical_features]
    mns = MinMaxScaler()
    X_scaled = mns.fit_transform(X_numerical)
    n_components = min(X_scaled.shape[0], X_scaled.shape[1], 3)  # Adjust dynamically
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, df['Titles']

def cosine_sim(query, index, X_pca):
    cosine = []
    query_vec = np.array(X_pca[query, :])
    for i in range(len(index[0])):
        temp_vec = np.array(X_pca[index[0][i], :])
        score = dot(query_vec, temp_vec) / (norm(query_vec) * norm(temp_vec))
        cosine.append(score)
    return cosine

def recommend(title, df, X_pca):
    NN = NearestNeighbors(algorithm='brute', metric='cosine')
    NN.fit(X_pca)
    idx = df[df['Titles'] == title].index[0]
    distances, index = NN.kneighbors(X_pca[idx].reshape(1, -1), n_neighbors=6)
    cosine_scores = cosine_sim(idx, index, X_pca)
    recommendations = df.iloc[index[0][1:]]['Titles'].tolist()
    return recommendations

st.title("Netflix Recommendation System")
df = load_data()
X_pca, titles = preprocess_data(df)
selected_title = st.selectbox("Select a TV Show", titles)
if st.button("Get Recommendations"):
    results = recommend(selected_title, df, X_pca)
    st.write("### Recommended Shows:")
    for rec in results:
        st.write(f"- {rec}")

