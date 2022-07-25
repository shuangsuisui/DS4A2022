import numpy as np
import pandas as pd

# MovieID, Title, Genres
movie_data = pd.DataFrame([movie.replace('\n','').split('::') for movie in open('movies.txt', encoding="ISO-8859-1").readlines()],
                          columns=['MovieID', 'Title', 'Genres'])

# UserID, Gender, Age, Occupation, Zip-code
user_data = pd.DataFrame([user.replace('\n','').split('::') for user in open('users.txt', encoding="ISO-8859-1").readlines()],
                         columns=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
user_data['Age'] = user_data['Age'].astype(int)
# user_validation = np.random.choice(user_data['UserID'], 5)

# UserID, MovieID, Rating, Timestamp
rating_data = pd.DataFrame([rating.replace('\n','').split('::') for rating in open('ratings.txt', encoding="ISO-8859-1").readlines()],
                           columns=['UserID', 'MovieID', 'Rating', 'Timestamp'])
rating_data['Rating'] = rating_data['Rating'].astype(int)
rating_data['Timestamp'] = pd.to_datetime(rating_data['Timestamp'], unit='s')
rating_data['Rank_Latest'] = rating_data.groupby(['UserID'])['Timestamp'].rank(method='first',ascending=False)

# dataset 3: validation user set
user_validation = ['5985' '3280' '5126' '907' '106']
# dataset 1: training data set # 994169 6040
training_rating = rating_data[rating_data['Rank_Latest'] != 1][~rating_data['UserID'].isin(user_validation)]
# dataset 2: testing data set
testing_rating = rating_data[rating_data['Rank_Latest'] == 1][~rating_data['UserID'].isin(user_validation)]