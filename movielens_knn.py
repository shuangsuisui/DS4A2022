import math
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.stats import mode
from sklearn.metrics import mean_squared_error

# MovieID, Title, Genres
movie_data = pd.DataFrame([movie.replace('\n','').split('::') for movie in open('movies.txt', encoding="ISO-8859-1").readlines()],
                          columns=['MovieID', 'Title', 'Genres'])
movie_data['MovieID'] = movie_data['MovieID'].astype(int)

# UserID, Gender, Age, Occupation, Zip-code
user_data = pd.DataFrame([user.replace('\n','').split('::') for user in open('users.txt', encoding="ISO-8859-1").readlines()],
                         columns=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']).drop(['Zip-code'], axis=1)
user_data['UserID'] = user_data['UserID'].astype(int)
user_data['Age'] = user_data['Age'].astype(int)
user_data = user_data.replace([1,18,25,35,45,50,56], [1,2,3,4,5,6,7])  # {1: 1, 18: 2, 25: 3, 35: 4, 45: 5, 50: 6, 56:  7}
user_data = user_data.replace(['F', 'M'], [0, 1]) # {'F': 0, 'M': 1}

# UserID, MovieID, Rating, Timestamp
rating_data = pd.DataFrame([rating.replace('\n','').split('::') for rating in open('ratings.txt', encoding="ISO-8859-1").readlines()],
                           columns=['UserID', 'MovieID', 'Rating', 'Timestamp'])
rating_data['UserID'] = rating_data['UserID'].astype(int)
rating_data['MovieID'] = rating_data['MovieID'].astype(int)
rating_data['Rating'] = rating_data['Rating'].astype(int)
rating_data['Timestamp'] = pd.to_datetime(rating_data['Timestamp'], unit='s')
rating_data['Rank_Latest'] = rating_data.groupby(['UserID', 'Rating'])['Timestamp'].rank(method='first',ascending=False)

# training dataset
training = rating_data[rating_data['Rank_Latest'] != 1].drop(['Rank_Latest'], axis=1).reset_index()
# testing dataset
testing = rating_data[rating_data['Rank_Latest'] == 1].drop(['Rank_Latest'], axis=1).reset_index()
# validation dataset
validation = [2484, 4448, 2106, 5702, 1018]

# pivot table
rate_pivot = training.pivot(index='MovieID', columns='UserID', values='Rating').fillna(0)

# get all ratings for each movie
movie_rating = training.groupby('MovieID').agg(RatingCount = pd.NamedAgg(column='Rating', aggfunc='count'),
                                                  RatingAve = pd.NamedAgg(column='Rating', aggfunc='mean')).reset_index()
movie_rating['Popularity'] = movie_rating['RatingCount'] / movie_rating['RatingCount'].max() # * 10
# movie_rating['RatingAve'] *= 10
movie_rating = pd.merge(movie_data.drop(['Title', 'Genres'], axis=1), movie_rating.drop(['RatingCount'], axis=1), how='left', on='MovieID').fillna(0)

# merge genres information
categories = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
              'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movie_genres = {category: [] for category in categories}
for genres in movie_data['Genres']:
    genre_dict = {category: 0 for category in categories}
    for genre in genres.split('|'):
        genre_dict[genre] += 1
    for key in movie_genres.keys():
        if genre_dict[key] == 1:
            movie_genres[key].append(5)
        else:
            movie_genres[key].append(0)
for key in movie_genres.keys():
    movie_rating[key] = movie_genres[key]

movie_info = pd.merge(movie_rating, rate_pivot, how='left', on='MovieID').fillna(0)
knn = NearestNeighbors(n_neighbors=6, algorithm = 'brute', metric = 'euclidean')

# RMSE
# prediction = []
# for itr in range(len(testing)): #
#     print(itr)
#     userid = testing['UserID'][itr]
#     movieid = testing['MovieID'][itr]
#     watched_list = list(training[training['UserID'] == userid]['MovieID']) + [movieid]
#     movie_info_selected = movie_info[movie_info['MovieID'].isin(watched_list)]
#     movie_index_list = list(movie_info_selected['MovieID'])
#     movie_info_selected = movie_info_selected.set_index('MovieID')
#     neighbors = knn.fit(movie_info_selected)
#     dist, ind = knn.kneighbors(movie_info_selected.loc[movieid].values.reshape(1, -1), n_neighbors=6)
#     neighbor_rating = [rate_pivot[userid][movie_index_list[ind.flatten()[j]]] for j in range(1, len(dist.flatten()))]
#     neighbor_dist = [ind.flatten()[j] for j in range(1, len(dist.flatten()))]
#     # prediction.append(mode([r for r in neighbor_rating if r != 0])[0][0]) # mode
#     # prediction.append(np.round(np.average([r for r in neighbor_rating if r != 0])))  # average
#     prediction.append(np.round(np.average(neighbor_rating, weights=neighbor_dist)))  # weighted average
#
# print(prediction)
# rmse = math.sqrt(mean_squared_error(list(testing['Rating']), prediction))
# print(rmse)

# 5 neighbors scaled with weighted average, rmse = 1.45720288696846
# 5 neighbors with mode, rmse = 1.62
# 5 neighbors with average, rmse = 1.45720288696846
# 10 neighbors with average, rmse = 1.4642009373421536
# 10 neighbors scaled with weighted average, rmse = 1.4537639695764346

# individual recommendation
rate_pivot = training.pivot(index='UserID', columns='MovieID', values='Rating').fillna(0)
user_rating = training.groupby('UserID').agg(RatingCount = pd.NamedAgg(column='Rating', aggfunc='count'),
                                             RatingAve = pd.NamedAgg(column='Rating', aggfunc='mean')).reset_index()
user_rating['RatingCount'] = user_rating['RatingCount'] / user_rating['RatingCount'].max()
user_rating = pd.merge(user_data.drop(['Occupation'], axis=1), user_rating.drop(['RatingCount'], axis=1), how='left', on='UserID').fillna(0)
occupations = [str(i) for i in range(21)]
user_occupation = {occupation: [] for occupation in occupations}
for occupation in user_data['Occupation']:
    for key in user_occupation.keys():
        if key == occupation:
            user_occupation[key].append(1)
        else:
            user_occupation[key].append(0)
for key in user_occupation.keys():
    user_rating[key] = user_occupation[key]
user_info = pd.merge(user_rating, rate_pivot, how='left', on='UserID').fillna(0)
user_idx =  user_info['UserID']
user_info = user_info.set_index('UserID')
for userid in validation: #
    neighbors = knn.fit(user_info)
    dist, ind = knn.kneighbors(user_info.loc[int(userid)].values.reshape(1, -1), n_neighbors=6)
    neighbor_user = ind[0][1:]
    watched_list = list(training[training['UserID'] == int(userid)]['MovieID'])
    list_of_movie = training[training['UserID'].isin(neighbor_user)][rating_data['Rating'] == 5][~rating_data['MovieID'].isin(watched_list)]['MovieID']
    recommendation = list(movie_info[movie_info['MovieID'].isin(list_of_movie)].sort_values(by=['RatingAve']).head(5)['MovieID'])
    print('Recommendation for User ', userid, ' : ', recommendation)

# Recommendation for User  2484  :  [3564, 3946, 2713, 3354, 3752]
# Recommendation for User  4448  :  [208, 2190, 1841, 74, 1006]
# Recommendation for User  2106  :  [1389, 193, 2450, 1499, 3710]
# Recommendation for User  5702  :  [3593, 519, 3689, 737, 2450]
# Recommendation for User  1018  :  [2450, 2113, 1499, 2642, 360]