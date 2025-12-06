from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

movies = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('model.pkl', 'rb'))

def recommend(movie):
    if movie not in movies['title'].values:
        return []

    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommendations = []
    for i in movie_list:
        recommendations.append(movies.iloc[i[0]].title)

    return recommendations


@app.route('/', methods=['GET', 'POST'])
def home():
    all_movies = movies['title'].values

    if request.method == 'POST':
        movie_name = request.form['movie_name']
        recs = recommend(movie_name)
        return render_template('result.html', movie_name=movie_name, recs=recs)

    return render_template('index.html', all_movies=all_movies)


if __name__ == '__main__':
    app.run(debug=True)
