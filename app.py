
import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide"
)

@st.cache_resource
def load_data():
    movies = joblib.load("movies.pkl")
    tfidf_matrix = joblib.load("tfidf_matrix.pkl")
    return movies, tfidf_matrix

movies, tfidf_matrix = load_data()

st.title("🎬 CineMatch")
st.subheader("AI Movie Recommendation System")
st.write("Discover movies similar to your favourites!")

movie_list = sorted(movies["title"].dropna().unique())

selected_movie = st.selectbox(
    "Choose your favourite movie:",
    movie_list
)

number = st.slider(
    "Number of recommendations",
    min_value=5,
    max_value=20,
    value=10
)

if st.button("Recommend Movies"):
    matches = movies[
        movies["title"].str.lower() == selected_movie.lower()
    ]

    if not matches.empty:
        movie_index = matches.index[0]

        scores = linear_kernel(
            tfidf_matrix[movie_index],
            tfidf_matrix
        ).flatten()

        similar_indices = scores.argsort()[::-1][1:number+1]

        st.success("Here are your recommended movies!")

        for index in similar_indices:
            movie = movies.iloc[index]

            st.markdown("---")
            st.subheader(f"🎥 {movie['title']}")

            st.write(
                "**Rating:**",
                movie["vote_average"],
                "/ 10"
            )

            genres = movie["genres"]

            if isinstance(genres, list):
                genres = ", ".join(genres)

            st.write("**Genres:**", genres)

    else:
        st.warning("Movie not found.")

st.markdown("---")
st.caption("CineMatch | AI Movie Recommendation System")
