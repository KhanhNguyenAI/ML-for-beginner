import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import re

df = pd.read_csv(r'C:\Users\97ngu\OneDrive\Desktop\course\ML-EX\chua lam\Recommendation_System\movie_data\movies.csv', sep='\t', encoding='windows-1252')
df['genres'] = df['genres'].str.replace('|', ' ').str.replace('-', '')
def recommend_movies(use_movie, top_n): 
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), stop_words='english', lowercase=True)
    tfidf_matrix = vectorizer.fit_transform(df['genres'])   
    # Do NOT overwrite df, use a new variable for the TF-IDF DataFrame
    tfidf_df = pd.DataFrame(tfidf_matrix.todense(), index=df['title'], columns=vectorizer.get_feature_names_out())
    print(tfidf_df)
    sim = cosine_similarity(tfidf_matrix)
    sim_df = pd.DataFrame(sim, index=df['title'], columns=df['title'])
    sorted_sim_df = sim_df.sort_values(by=use_movie, ascending=False)
    top_n_similar = sorted_sim_df[use_movie].head(top_n + 1)  # +1 to skip the movie itself
    movies = [movie for movie in top_n_similar.keys() if movie != use_movie]
    print(top_n_similar)
    def extract_year(title):
        match = re.search(r'\((\d{4})\)', title)
        return int(match.group(1)) if match else 0
    movies_sorted = sorted(movies, key=extract_year, reverse=True)
    return movies_sorted

use_movie = 'Drunks (1997)'
top_n = 10
recommended_movies = recommend_movies(use_movie, top_n)
print(f"Top {top_n} movies similar to '{use_movie}':")
for movie in recommended_movies:
    print(movie)