import streamlit as st
import pickle
import requests

# Load data
movies = pickle.load(open('artifacts/movies.pkl','rb'))
similarity = pickle.load(open('artifacts/similarity.pkl','rb'))

# ---------- Poster Fetch ----------

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=YOUR_API_KEY"
    data = requests.get(url).json()
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']

# ---------- Recommendation Function ----------

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_posters = []

    for i in movie_list:
        movie_id = movies.iloc[i[0]].movie_id
        
        # Append title
        recommended_movies.append(movies.iloc[i[0]].title)
        
        # Append poster
        try:
            recommended_posters.append(fetch_poster(movie_id))
        except:
            recommended_posters.append("")

    return recommended_movies, recommended_posters

# ---------- UI ----------

st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox(
    "Choose a movie",
    movies['title'].values
)

if st.button('Recommend'):
    names, posters = recommend(selected_movie)

    col1, col2, col3, col4, col5 = st.columns(5)

    for col, name, poster in zip([col1,col2,col3,col4,col5], names, posters):
        with col:
            st.text(name)
            if poster:
                st.image(poster)
