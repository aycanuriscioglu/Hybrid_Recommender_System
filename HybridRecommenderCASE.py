movie = pd.read_csv('datasets/movie.csv')
rating = pd.read_csv('datasets/rating.csv')

df_ = movie.merge(rating, how="left", on="movieId")
df=df_.copy()
df.head()
df.isnull().sum()
df.describe()

comment_counts = pd.DataFrame(df["title"].value_counts())


rare_movies = comment_counts[comment_counts["title"] <= 1000].index 
common_movies = df[~df["title"].isin(rare_movies)] 

user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")

random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=2387463).values)
random_user

#############################################
# User-Based Recommendation
#############################################
random_user_df = user_movie_df[user_movie_df.index == random_user]

movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
len(movies_watched)
movies_watched

movies_watched_df = pd.DataFrame(user_movie_df[movies_watched])


user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId","movie_count"]

perc = len(movies_watched) * 60 / 100
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]


final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                      random_user_df[movies_watched]])

corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df = corr_df.reset_index()
corr_df

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]].reset_index(drop=True)

top_users = top_users.sort_values(by='corr', ascending=False)

top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()

movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False).head()
movies_to_be_recommend.merge(movie[["movieId", "title"]])

#############################################
# Item-Based Recommendation
#############################################

random_user
movie.head()
rating.head()
df.head()

movie_id = rating[(rating["userId"] == random_user) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]
movie_name = df[(df["userId"] == random_user) & (df["movieId"] == 1210)][["userId","title","timestamp","rating","movieId"]]
movie_name
movie_name = "Star Wars: Episode VI - Return of the Jedi (1983)"

movie_name = user_movie_df[movie_name]

recommendation_df=user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)
recommendation_df=user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)[1:6]
recommendation_df

