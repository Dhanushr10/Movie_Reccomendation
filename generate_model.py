import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
movies = pd.read_csv("movies.csv")

# Vectorize the text (overview)
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['overview'].values.astype('U')).toarray()

# Calculate similarity matrix
similarity = cosine_similarity(vectors)

# Save movies dataframe
pickle.dump(movies, open('movies.pkl', 'wb'))

# Save similarity matrix
pickle.dump(similarity, open('model.pkl', 'wb'))

print("movies.pkl and model.pkl created successfully!")
