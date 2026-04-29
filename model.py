import pandas as pd
import pickle
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Helper Functions ----------

def convert(text):
    """Convert stringified list of dicts to list of names"""
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

def fetch_director(text):
    """Extract director name from crew"""
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

def collapse(L):
    """Remove spaces between words"""
    return [i.replace(" ", "") for i in L]

# ---------- Load Data ----------

movies = pd.read_csv('data/movies.csv')
credits = pd.read_csv('data/credits.csv')

# Merge datasets
movies = movies.merge(credits, on='title')

# Select columns
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

# Drop missing values
movies.dropna(inplace=True)

# ---------- Feature Engineering ----------

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert)
movies['crew'] = movies['crew'].apply(fetch_director)

# Take top 3 cast
movies['cast'] = movies['cast'].apply(lambda x: x[:3])

# Remove spaces
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)
movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)

# Convert overview to list
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Create tags
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new_df = movies[['movie_id','title','tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

# ---------- Vectorization ----------

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# ---------- Similarity ----------

similarity = cosine_similarity(vectors)

# ---------- Save ----------

pickle.dump(new_df, open('artifacts/movies.pkl', 'wb'))
pickle.dump(similarity, open('artifacts/similarity.pkl', 'wb'))

print("✅ Model built and saved successfully!")
