import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load your CSV (or whatever dataset you used earlier)
df = pd.read_csv("your_movie_dataset.csv")   # change this line

# Example: assuming dataset has a 'title' and 'tags' column
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(df['tags']).toarray()
similarity = cosine_similarity(vectors)

# Save the required files
pickle.dump(df, open('movies.pkl', 'wb'))
pickle.dump(similarity, open('model.pkl', 'wb'))

print("movies.pkl and model.pkl created successfully!")
