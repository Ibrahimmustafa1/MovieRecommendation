from flask import Flask, jsonify
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load data and models
ratings = pd.read_csv('maped_df.csv')
df_movies = pd.read_csv('movies_one_hot.csv')
user_one_hot = joblib.load('./models/user_encoder.joblib')
model = tf.keras.models.load_model('./models/model5.h5')
moviesNames_df = pd.read_csv('movies.csv')
df_links = pd.read_csv('links.csv')

movies_ids = df_movies['itemID']
movies_features = df_movies.drop('itemID', axis=1).values

def get_tmdb_id(movie_id):
    tmdb_id = ratings.loc[ratings['movieId'] == movie_id, 'tmdbId'].values
    return tmdb_id[0] if len(tmdb_id) > 0 else None

@app.route('/rated_movies/<int:user_id>', methods=['GET'])
def get_rated_movies(user_id):
    user_ratings = ratings[ratings['userId'] == user_id].nlargest(10, 'rating')
    if user_ratings.shape[0] < 10:
        user_ratings = ratings[ratings['rating'] == user_id]
    user_ratings = user_ratings.to_dict(orient='records')
    return jsonify(user_ratings)

@app.route('/predict/<int:user_id>', methods=['GET'])
def predict(user_id):
    user_vec = user_one_hot.transform([[user_id]]).toarray()
    user_vec_replicated = np.repeat(user_vec, len(movies_ids), axis=0)
    pred_rates = model.predict([user_vec_replicated, movies_features]).flatten()

    df_rates = pd.DataFrame({'movie_id': movies_ids, 'rate': pred_rates})
    df_rates_sorted = df_rates.nlargest(20, 'rate')
    
    top_10 = df_rates_sorted.to_dict(orient='records')
    lowest_10 = df_rates_sorted.nsmallest(20, 'rate').to_dict(orient='records')
    
    for movie in top_10 + lowest_10:
        movie_id = movie['movie_id']
        tmdb_id = get_tmdb_id(movie_id)
        movie['tmdb_id'] = tmdb_id
    
    top_and_lowest = {'high': top_10, 'low': lowest_10}
    return jsonify(top_and_lowest)

@app.route('/movies', methods=['GET'])
def get_movies_names():
    df_titles = pd.concat([moviesNames_df, df_links['tmdbId']], axis=1).dropna()
    return jsonify(df_titles.to_dict(orient='records'))

@app.route('/similar/<int:item_id>', methods=['GET'])
def recommend_similar_items(item_id):
    item_features = df_movies.loc[df_movies['itemID'] == item_id].drop('itemID', axis=1).values.reshape(1, -1)
    item_embedding = model.layers[3](item_features)
    
    all_item_features = df_movies.drop('itemID', axis=1).values
    all_item_embeddings = model.layers[3](all_item_features)
    
    similarities = cosine_similarity(item_embedding, all_item_embeddings)
    similar_indices = np.argsort(similarities.flatten())[:-11:-1]
    
    top_similar_items = []
    for idx in similar_indices:
        sim_item_id = df_movies.iloc[idx]['itemID']
        title = moviesNames_df.loc[moviesNames_df['movieId'] == sim_item_id, 'title'].values[0]
        tmdb_id = ratings.loc[ratings['movieId'] == sim_item_id, 'tmdbId'].values[0]
        top_similar_items.append({'movie_id': sim_item_id, 'tmdb_id': tmdb_id, 'title': title})
    
    return jsonify(top_similar_items)

if __name__ == '__main__':
    app.run(debug=True)
