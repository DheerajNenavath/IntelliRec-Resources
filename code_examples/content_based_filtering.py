
# Content-Based Filtering using Cosine Similarity
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample dataset
df = pd.DataFrame({
    'title': ['The Matrix', 'Inception', 'Interstellar', 'The Dark Knight'],
    'description': [
        'Sci-fi action movie with AI theme',
        'Mind-bending thriller with dream layers',
        'Sci-fi film about space and time',
        'Superhero film with deep moral conflict']
})

# TF-IDF feature matrix
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['description'])

# Cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommend movies similar to 'Inception'
index = df[df['title'] == 'Inception'].index[0]
similarities = list(enumerate(cosine_sim[index]))
similar_movies = sorted(similarities, key=lambda x: x[1], reverse=True)[1:3]

print("Movies similar to Inception:")
for i in similar_movies:
    print("-", df.iloc[i[0]]['title'])
