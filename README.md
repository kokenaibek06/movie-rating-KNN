Movie Recommendation System using KNN

This project is a movie recommendation system built with Python and Machine Learning.
The main goal of this project is to recommend similar movies based on user rating behavior.
The system uses collaborative filtering and K-Nearest Neighbors (KNN) with cosine distance to find movies that are similar to each other.
Project Overview
In this project, I used movie rating data to build a recommendation system.
Each movie is represented as a vector of user ratings. Then, the KNN algorithm compares these vectors and finds movies with similar rating patterns.
For example, if users who liked one movie also gave similar ratings to another movie, the system will consider these movies similar.
Dataset
The dataset contains movie ratings with columns such as:
userId
movieId
rating
title
The movie and rating datasets were merged using movieId.
df = pd.merge(movies_df, rating_df, on='movieId')
After merging, each row represents a user rating for a specific movie.
Data Preprocessing
First, I checked missing values in the movie title column:
df['title'].isna().sum()
Then, I calculated how many ratings each movie received:
mov_rating_count = (
    df.groupby(by=['title'])['rating']
    .count()
    .reset_index()
    .rename(columns={'rating': 'totalRatingCount'})
    [['title', 'totalRatingCount']]
)
After that, I merged the rating count back into the main dataframe:
rating_with_totalRatingCount = df.merge(
    mov_rating_count,
    on='title',
    how='left'
)
This step adds a new column called totalRatingCount, which shows how many times each movie was rated.
Filtering Popular Movies
Movies with very few ratings can give weak recommendations.
So, I filtered movies based on a popularity threshold.
popularity_threshold = 50

rating_popular_movie = rating_with_totalRatingCount[
    rating_with_totalRatingCount['totalRatingCount'] >= popularity_threshold
]
This helps the model focus only on movies that have enough rating data.
Creating the Movie-User Matrix
To use KNN, I created a pivot table where:
rows are movie titles
columns are user IDs
values are ratings
movie_matrix = rating_popular_movie.pivot_table(
    index='title',
    columns='userId',
    values='rating'
).fillna(0)
Missing values were filled with 0, meaning the user did not rate that movie.
Example structure:
title	user 1	user 2	user 3
Toy Story	4.0	0	5.0
Jumanji	0	3.5	4.0
Sparse Matrix
The movie-user matrix contains many zeros because most users rate only a small number of movies.
To save memory and improve performance, I converted the matrix into a sparse matrix using csr_matrix.
from scipy.sparse import csr_matrix

movie_features_matrix = csr_matrix(movie_matrix.values)
Model Training
I used the NearestNeighbors model from Scikit-learn.
from sklearn.neighbors import NearestNeighbors

model_knn = NearestNeighbors(
    metric='cosine',
    algorithm='brute'
)

model_knn.fit(movie_features_matrix)
Why Cosine Distance?
Cosine distance measures similarity between two movie rating vectors.
distance close to 0 means movies are very similar
distance close to 1 means movies are less similar
This is useful for recommendation systems because it compares rating patterns between movies.
Why Brute Force Algorithm?
The brute force algorithm compares one movie with all other movies and finds the nearest neighbors.
It is simple and works well for this type of project, especially with cosine similarity and sparse matrices.
Getting Recommendations
A random movie is selected from the movie matrix:
query_index = np.random.choice(movie_matrix.shape[0])
print(query_index)
Then, the model finds the nearest movies:
distances, indices = model_knn.kneighbors(
    movie_features_matrix[query_index],
    n_neighbors=6
)
I used n_neighbors=6 because the first result is usually the same movie.
The next 5 movies are the actual recommendations.
Recommendation Output
for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for {0}:\n'.format(
            movie_matrix.index[query_index]
        ))
    else:
        print('{0}: {1}, with distance of {2}:'.format(
            i,
            movie_matrix.index[indices.flatten()[i]],
            distances.flatten()[i]
        ))
Example output:
Recommendations for Toy Story (1995):

1: Aladdin (1992), with distance of 0.35
2: Lion King, The (1994), with distance of 0.38
3: Jurassic Park (1993), with distance of 0.42
4: Back to the Future (1985), with distance of 0.45
5: Jumanji (1995), with distance of 0.47
Technologies Used
Python
Pandas
NumPy
SciPy
Scikit-learn
Jupyter Notebook
Machine Learning Concepts Used
Recommendation System
Collaborative Filtering
K-Nearest Neighbors
Cosine Distance
Sparse Matrix
User-Item Matrix
Data Preprocessing
Project Goal
The main goal of this project is to practice building a recommendation system using real rating data.
This project helped me understand how collaborative filtering works and how KNN can be used to recommend similar items.
