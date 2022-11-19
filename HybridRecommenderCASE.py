
#############################################
# PROJE: Hybrid Recommender System
#############################################

# ID'si verilen kullanıcı için item-based ve user-based recomennder yöntemlerini kullanarak tahmin yapınız.
# 5 öneri user-based modelden 5 öneri de item-based modelden ele alınız ve nihai olarak 10 öneriyi 2 modelden yapınız.


#VERİ SETİ HİKAYESİ
#Veri seti, bir film tavsiye hizmeti olan MovieLens tarafından sağlanmıştır. İçerisinde filmler
#ile birlikte bu filmlere yapılanderecelendirme puanlarını barındırmaktadır. 27.278 filmde 2.000.0263
#derecelendirme içermektedir. Bu veri seti ise 17 Ekim 2016 tarihinde oluşturulmuştur. 138.493
#kullanıcı ve 09 Ocak 1995 ile 31 Mart 2015 tarihleri arasında verileri içermektedir. Kullanıcılar
#rastgele seçilmiştir. Seçilen tüm kullanıcıların en az 20 filme oy verdiği bilgisi mevcuttur.


#############################################
# Görev 1: Verinin Hazırlanması
#############################################
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



# Adım 1: Movie ve Rating veri setlerini okutunuz.
# movieId, film adı ve filmin tür bilgilerini içeren veri seti
movie = pd.read_csv('datasets/movie.csv')
rating = pd.read_csv('datasets/rating.csv')

# UserID, film adı, filme verilen oy ve zaman bilgisini içeren veri seti

# Adım 2: Rating veri setine filmlerin isimlerini ve türünü movie film setini kullanrak ekleyiniz.
# Ratingdeki kullanıcıların oy kullandıkları filmlerin sadece id'si var.
# Idlere ait film isimlerini ve türünü movie veri setinden ekliyoruz.

df_ = movie.merge(rating, how="left", on="movieId")
df=df_.copy()
df.head()
df.isnull().sum()
df.describe()


# Adım 3: Herbir film için toplam kaç kişinin oy kullandığını hesaplayınız.Toplam oy kullanılma sayısı 1000'un altında
# olan filmleri veri setinden çıkarınız.
# Herbir film için toplam kaç kişinin oy kullanıldığını hesaplıyoruz.

df["title"].count()
df["title"].nunique()
df["title"].value_counts().head()
comment_counts = pd.DataFrame(df["title"].value_counts())
comment_counts.shape

# Toplam oy kullanılma sayısı 1000'in altında olan filmlerin isimlerini rare_movies de tutuyoruz.
# Ve veri setinden çıkartıyoruz

rare_movies = comment_counts[comment_counts["title"] <= 1000].index #24103 film 1000'den az puana sahip
common_movies = df[~df["title"].isin(rare_movies)] #rarein içinde olmayanları df içine girme
common_movies.shape #17766015 toplam oy var
common_movies["title"].nunique() #3159 toplam film kaldı
df["title"].nunique()  #27262 setteki ilk film sayısı
common_movies.head()

# Adım 4: # index'te userID'lerin sutunlarda film isimlerinin ve değer olarakta ratinglerin bulunduğu
# dataframe için pivot table oluşturunuz.

user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
user_movie_df.head()

# Adım 5: Yukarıda yapılan tüm işlemleri fonksiyonlaştıralım

def create_user_movie_df(DataFrame):
    import pandas as pd
    movie = pd.read_csv('datasets/movie.csv')
    rating = pd.read_csv('datasets/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df


create_user_movie_df(df)

#############################################
# Görev 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
#############################################

# Adım 1: Rastgele bir kullanıcı id'si seçiniz.

random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=2387463).values)
random_user

# Adım 2: Seçilen kullanıcıya ait gözlem birimlerinden oluşan random_user_df adında yeni bir dataframe oluşturunuz.

random_user_df = user_movie_df[user_movie_df.index == random_user]
random_user_df

# Adım 3: Seçilen kullanıcının oy kullandığı filmleri movies_watched adında bir listeye atayınız.

movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
len(movies_watched)
movies_watched

#############################################
# Görev 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
#############################################

# Adım 1: Seçilen kullanıcının izlediği fimlere ait sutunları user_movie_df'ten seçiniz ve movies_watched_df adında
# yeni bir dataframe oluşturuyoruz.

movies_watched_df = pd.DataFrame(user_movie_df[movies_watched])
movies_watched_df

# Adım 2: Herbir kullancının seçili user'in izlediği filmlerin kaçını izlediği bilgisini taşıyan user_movie_count
# adında yeni bir dataframe oluşturunuz.
# Ve yeni bir df oluşturuyoruz.

user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count.describe()
user_movie_count.head()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId","movie_count"]
user_movie_count

# Adım 3: Seçilen kullanıcının oy verdiği filmlerin yüzde 60 ve üstünü izleyenleri benzer kullanıcılar olarak
# görüyoruz.
# Bu kullanıcıların id’lerinden users_same_movies adında bir liste oluşturunuz.

perc = len(movies_watched) * 60 / 100
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]
users_same_movies.shape

#############################################
# Görev 4: Öneri Yapılacak Kullanıcı ile En Benzer Kullanıcıların Belirlenmesi
#############################################

# Adım 1: user_same_movies listesi içerisindeki seçili user ile benzerlik gösteren kullanıcıların
# id’lerinin bulunacağı şekilde movies_watched_df dataframe’ini filtreleyiniz.

final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                      random_user_df[movies_watched]])
final_df.head()

# Adım 2: Kullanıcıların birbirleri ile olan korelasyonlarının bulunacağı yeni bir corr_df dataframe’i oluşturunuz.

corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
corr_df.head()
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df = corr_df.reset_index()
corr_df

#corr_df
# f[corr_df["user_id_1"] == random_user]

# Adım 3: Seçili kullanıcı ile yüksek korelasyona sahip (0.65’in üzerinde olan) kullanıcıları
# filtreleyerek top_users adında yeni bir dataframe oluşturunuz.

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]].reset_index(drop=True)

top_users.head()

top_users = top_users.sort_values(by='corr', ascending=False)
top_users.head()


top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

# Adım 4:  top_users dataframe’ine rating veri seti ile merge ediniz


top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
top_users_ratings.head()
top_users_ratings.shape
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]
top_users_ratings.head()

#############################################
# Görev 5: Weighted Average Recommendation Score'un Hesaplanması ve İlk 5 Filmin Tutulması
#############################################

# Adım 1: Her bir kullanıcının corr ve rating değerlerinin çarpımından oluşan weighted_rating adında yeni bir değişken oluşturunuz.

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
top_users_ratings.head()

# Adım 2: Film id’siv e her bir filme ait tüm kullanıcıların weighted rating’lerinin ortalama değerini içeren
# recommendation_df adında yeni bir
# dataframe oluşturunuz.

recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df.head()
recommendation_df = recommendation_df.reset_index()
recommendation_df["weighted_rating"].max()
recommendation_df["weighted_rating"].describe()

# Adım 3: Adım3: recommendation_df içerisinde weighted rating'i 3.5'ten büyük olan filmleri seçiniz ve
# weighted rating’e göre sıralayınız.
# İlk 5 gözlemi movies_to_be_recommend olarak kaydediniz.

movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False).head()
movies_to_be_recommend

# Adım 4:  Tavsiye edilen 5 filmin isimlerini getiriniz.

movies_to_be_recommend.merge(movie[["movieId", "title"]])

#############################################
# Adım 6: Item-Based Recommendation
#############################################

# Kullanıcının en son izlediği ve en yüksek puan verdiği filmin adına göre item-based öneri yapınız.
random_user

# Adım 1: movie,rating veri setlerini okutunuz.

movie.head()
rating.head()
df.head()

# Adım 2: Öneri yapılacak kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin id'sinin alınız.

movie_id = rating[(rating["userId"] == random_user) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]
movie_id

# Adım 3 :User based recommendation bölümünde oluşturulan user_movie_df dataframe’ini seçilen film id’sine göre filtreleyiniz.

movie_name = df[(df["userId"] == random_user) & (df["movieId"] == 1210)][["userId","title","timestamp","rating","movieId"]]
movie_name
movie_name = "Star Wars: Episode VI - Return of the Jedi (1983)"

# Adım 4: Filtrelenen dataframe’i kullanarak seçili filmle diğer filmlerin korelasyonunu bulunuz ve sıralayınız.

movie_name = user_movie_df[movie_name]
movie_name

recommendation_df=user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)
recommendation_df

# Adım 5: Seçili film’in kendisi haricinde ilk 5 film’I öneri olarak veriniz.

recommendation_df=user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)[1:6]
recommendation_df



""""EK ÇALIŞMA"""
import pandas as pd
from surprise import Reader, SVD, Dataset, accuracy
from surprise.model_selection import GridSearchCV, train_test_split, cross_validate


movie_ids = [131254, 131258, 356, 541, 1, 4422, 1210]
movies = ["Kein Bund für's Leben (2007)",
          "The Pirates (2014)",
          "Forrest Gump (1994)",
          "Blade Runner (1982)",
          "Toy Story (1995)"
          "Cries and Whispers (Viskningar och rop) (1972)",
          "Star Wars: Episode VI - Return of the Jedi (1983)"]


sample_df = df[df.movieId.isin(movie_ids)]
sample_df.head()

sample_df.shape

user_movie_df = sample_df.pivot_table(index=["userId"],
                                      columns=["title"],
                                      values="rating")

user_movie_df.shape

reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(sample_df[['userId',
                                       'movieId',
                                       'rating']], reader)

#model kurma
trainset, testset = train_test_split(data, test_size=.25)
svd_model = SVD()   #model nesnesi
svd_model.fit(trainset)
predictions = svd_model.test(testset)
predictions

#ortalama ne kadar hata yapıldığını öğrenmek için;
# accuracy.rmse(predictions)

sample_df[sample_df["userId"] == 71265]
svd_model.predict(uid=71265, iid=356, verbose=True)

sample_df[sample_df["userId"] == 10906]
svd_model.predict(uid=10906, iid=1, verbose=True)

sample_df[sample_df["userId"] == 4459]
svd_model.predict(uid=4459, iid=1210, verbose=True)


# Adım 3: Model Tuning
##############################
#modelin tahmin oranını arttırma ve dışsal parametrelerini optimize etme

param_grid = {'n_epochs': [5, 10, 20, 50, 75],
              'lr_all': [0.001, 0.002, 0.005, 0.007, 0.090]}


gs = GridSearchCV(SVD,
                  param_grid,
                  measures=['rmse', 'mae'],
                  cv=3,
                  n_jobs=-1,
                  joblib_verbose=True)

gs.fit(data)

gs.best_score['rmse']#önceki orana göre daha iyi oran verdi
gs.best_params['rmse']#sonucu veren en iyi parametreler


# Final Model ve Tahmin
##############################
#daha iyi sonuca ulaşılacak parametreleri keşfettikten sonra model kurma işlemi bu parametreler kullanılarak
# tekrar yapılır. SVDnin ön tanımlı değerleri değiştirilir

dir(svd_model)
svd_model.n_epochs


#best paramdan gelen parametreler girileceği için
svd_model = SVD(**gs.best_params['rmse'])

data = data.build_full_trainset()
svd_model.fit(data)


sample_df[sample_df["userId"] == 71265]
svd_model.predict(uid=71265, iid=356, verbose=True)

sample_df[sample_df["userId"] == 10906]
svd_model.predict(uid=10906, iid=1, verbose=True)

sample_df[sample_df["userId"] == 4459]
svd_model.predict(uid=4459, iid=1210, verbose=True)
