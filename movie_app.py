import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# To run this app in command line streamlit run c:/Users/omkar/OneDrive/Desktop/Datascience/ML/movie_app.py


# Load the data
@st.cache_data
def load_data():
    # df = pd.read_csv(r"C:\Users\omkar\OneDrive\Desktop\Datasets\movies.csv")
    # df = pd.read_csv("movies.csv")
    df = pd.read_csv(r"C:\Users\omkar\OneDrive\Desktop\Datascience\ML\Movie_app\movies.csv")
    df['genres'].fillna('', inplace=True)
    return df

df = load_data()

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['genres'])

# Function to get the closest matching movie title
def close_match(movie_name):
    matches = difflib.get_close_matches(movie_name, df['title'].tolist())
    return matches[0] if matches else None

# Recommendation function
def recommend_movies(movie_title):
    movie_name = close_match(movie_title)
    if not movie_name:
        return None, []

    movie_index = df[df['title'] == movie_name].index[0]
    similarity_scores = cosine_similarity(tfidf_matrix[movie_index], tfidf_matrix)
    top_indexes = similarity_scores.argsort()[0][-11:-1][::-1]  # Exclude the movie itself
    recommended_titles = df.iloc[top_indexes]['title'].values.tolist()
    return movie_name, recommended_titles

# Streamlit UI
st.title("🎬 Movie Recommendation App")

st.markdown("""
Type in a movie name, and we'll recommend 10 similar movies based on genre similarity using **TF-IDF** and **Cosine Similarity**.
""")

user_input = st.text_input("Enter a movie name:")

if st.button("Recommend"):
    if user_input:
        matched_title, recommendations = recommend_movies(user_input)
        if recommendations:
            st.success(f"Movies similar to '{matched_title}':")
            for i, title in enumerate(recommendations, 1):
                st.write(f"{i}. {title}")
        else:
            st.error("Movie not found. Try a different name.")
    else:
        st.warning("Please enter a movie name.")
