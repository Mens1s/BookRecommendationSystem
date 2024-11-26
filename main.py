import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

books = pd.read_csv('datasets/Books.csv')
ratings = pd.read_csv('datasets/Ratings.csv')
users = pd.read_csv('datasets/Users.csv')

books.loc[187689, 'Book-Author'] = 'Downes, Larissa Anne'

users.drop(columns=['Age'], inplace=True)

books.loc[209538, 'Book-Author'] = 'Michael Teitelbbaum'
books.loc[209538, 'Book-Title'] = 'DK Readers: The Story of the X-Men, How It All Began (Level 4: Proficient Readers)'
books.loc[209538, 'Year-Of-Publication'] = 2000
books.loc[209538, 'Publisher'] = 'DK Publishing Inc'

books.loc[220731, 'Book-Title'] = "Peuple du ciel, suivi de 'Les Bergers'"
books.loc[220731, 'Book-Author'] = 'Jean-Marie Gustave Le Clézio'
books.loc[220731, 'Year-Of-Publication'] = 1990
books.loc[220731, 'Publisher'] = 'Gallimard'

books.loc[221678, 'Book-Title'] = 'DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)'
books.loc[221678, 'Book-Author'] = 'James Buckley'
books.loc[221678, 'Year-Of-Publication'] = 2000
books.loc[221678, 'Publisher'] = 'DK Publishing Inc'

books['Year-Of-Publication'] = books['Year-Of-Publication'].astype('int64')

books['Year-Of-Publication'].value_counts().sort_index(ascending=False).iloc[:20]

books[books['Year-Of-Publication']>2021][['Book-Title','Year-Of-Publication','Publisher','Book-Author']]

books.loc[37487, 'Year-Of-Publication'] = 1991
books.iloc[37487]

books.loc[37487, 'Year-Of-Publication'] = 1991

books.loc[55676, 'Year-Of-Publication'] = 2005

books.loc[37487, 'Book-Author'] = 'Bruce Coville'

books.loc[80264, 'Year-Of-Publication'] = 2003

books.loc[192993, 'Year-Of-Publication'] = 2003

books.loc[78168, 'Year-Of-Publication'] = 2001

books.loc[97826, 'Year-Of-Publication'] = 1981

books.loc[116053, 'Year-Of-Publication'] = 1995

books.loc[118294, 'Year-Of-Publication'] = 2023

books.loc[228173, 'Year-Of-Publication'] = 1987

books.loc[240169, 'Year-Of-Publication'] = 1996

books.loc[246842, 'Year-Of-Publication'] = 1925

books.loc[255409, 'Year-Of-Publication'] = 1937

books.loc[260974, 'Year-Of-Publication'] = 1991

books['Year-Of-Publication'].value_counts().sort_index(ascending=False).iloc[:20]

books[(books['Year-Of-Publication']<1400)&(books['Year-Of-Publication']>0)]

books_year_rational = books[books['Year-Of-Publication']!=0]['Year-Of-Publication'].value_counts().sort_index(ascending=False).iloc[:20]

books[books['Book-Author'].duplicated()]

author_book_count = books['Book-Author'].value_counts()

author_book_count = books[books['Book-Author']!= 'Not Applicable (Na )']
author_book_count_top50 = author_book_count.groupby('Book-Author').count()['Book-Title'].sort_values(ascending=False).head(50)

cool = sns.color_palette("cool", n_colors=len(author_book_count_top50.values))

publishers = books['Publisher'].value_counts()

top_50_publishers = publishers.sort_values(ascending=False)[:51]

cool = sns.color_palette("cool", n_colors=len(author_book_count_top50.values))

bookRating = pd.merge(ratings, books, on="ISBN")

bookRating.drop(columns=['Image-URL-S','Image-URL-M','Image-URL-L'],inplace=True)

averageRating = pd.DataFrame(bookRating.groupby('ISBN')['Book-Rating'].mean().round(1))
averageRating.reset_index(inplace=True)

averageRating.shape
averageRating.rename(columns={'Book-Rating':'Average-Rating'}, inplace=True)

averageRatingdf = pd.merge(bookRating, averageRating, on='ISBN')

averageRatingOnly = averageRatingdf[['ISBN','Average-Rating']]

averageRatingUnique = averageRatingOnly[['ISBN','Average-Rating']].drop_duplicates(subset=['ISBN'])

ratingBooks = pd.merge(books, averageRatingUnique, on='ISBN', how='inner')

books_with_rating = pd.merge(books, averageRatingUnique, on='ISBN')

books_with_rating = books_with_rating[['ISBN','Book-Title','Book-Author','Average-Rating','Year-Of-Publication','Publisher','Image-URL-S','Image-URL-M','Image-URL-L']]

books_with_rating.sort_values(by=['Average-Rating'], ascending=False).head(30)


ratings_sorted = books_with_rating['Average-Rating'].value_counts().sort_index(ascending=False)

top_20_ratings = books_with_rating['Average-Rating'].value_counts().drop(index=0.0).sort_values(ascending=False).head(20)

ratings_books_merged = ratings.merge(books, on='ISBN')

ratings_books_nonzero = ratings_books_merged[ratings_books_merged['Book-Rating']!=0]
num_rating_df = ratings_books_nonzero.groupby('Book-Title').count()['Book-Rating'].sort_values(ascending=False).reset_index()
num_rating_df.rename(columns={'Book-Rating':'Number-of-Ratings'}, inplace=True)

avg_rating_df = ratings_books_nonzero.groupby('Book-Title').mean(numeric_only=True)['Book-Rating'].reset_index()
avg_rating_df.rename(columns={'Book-Rating':'Average-Rating'}, inplace=True)

popularity_df = pd.merge(num_rating_df, avg_rating_df, on='Book-Title')

popularity_df_above_100 = popularity_df[popularity_df['Number-of-Ratings']>=100]
popularity_df_above_50 = popularity_df[popularity_df['Number-of-Ratings'] >= 50]
popularity_df_above_250 = popularity_df[popularity_df['Number-of-Ratings'] >= 250]
popularity_df_above_100.sort_values(by='Average-Rating', ascending=False).head()

# Defining a new function that can calculate the metric
def calcWeightedRating(row, avgRating, numOfRatings, minThres, defRating):
    weightedRating = ((row[avgRating] * row[numOfRatings]) + (minThres * defRating))/(row[numOfRatings] + minThres)
    return weightedRating

# For number of ratings above 100
popularity_df_above_100 = popularity_df_above_100.copy()
popularity_df_above_100['Weighted-Rating'] = popularity_df_above_100.apply(lambda x: calcWeightedRating(
     x, 'Average-Rating', 'Number-of-Ratings', 100, 5),axis=1)
popularity_df_above_100.sort_values(
    'Weighted-Rating', ascending=False).head(20)

# For number of ratings above 50
popularity_df_above_50 = popularity_df_above_50.copy()
popularity_df_above_50['Weighted-Rating'] = popularity_df_above_50.apply(lambda x: calcWeightedRating(
    x, 'Average-Rating', 'Number-of-Ratings', 50, 5), axis=1)
popularity_df_above_50.sort_values(
    'Weighted-Rating', ascending=False).head(20)

# For number of ratings above 250
popularity_df_above_250 = popularity_df_above_250.copy()
popularity_df_above_250['Weighted-Rating'] = popularity_df_above_250.apply(lambda x: calcWeightedRating(
    x, 'Average-Rating', 'Number-of-Ratings', 250, 5), axis=1)
popularity_df_above_250.sort_values(
    'Weighted-Rating', ascending=False).head(20)

popular_df_merge = pd.merge(popularity_df_above_100, books, on='Book-Title').drop_duplicates('Book-Title',keep='first')
popular_df_merge = popular_df_merge.drop(columns=['Image-URL-S', 'Image-URL-L'])
popular_df_merge.sort_values('Weighted-Rating', ascending=False).head(10)

users_ratings_count = ratings_books_merged.groupby('User-ID').count()['ISBN']
users_ratings_count = users_ratings_count.sort_values(ascending=False).reset_index()
users_ratings_count.rename(columns={'ISBN':'No-of-Books-Rated'}, inplace=True)

users_200 = users_ratings_count[users_ratings_count['No-of-Books-Rated']>=200]

books_with_users_200 = pd.merge(users_200, ratings_books_merged, on='User-ID')

books_ratings_count = ratings_books_merged.groupby('Book-Title').count()['ISBN'].sort_values(ascending=False).reset_index()
books_ratings_count.rename(columns={'ISBN':'Number-of-Book-Ratings'}, inplace=True)

books_ratings_50 = books_ratings_count[books_ratings_count['Number-of-Book-Ratings']>=50]

filtered_books = pd.merge(books_ratings_50, books_with_users_200,  on='Book-Title')

famous_books = filtered_books.groupby('Book-Title').count().reset_index()
famous_books = famous_books['Book-Title']
famous_books = books[books['Book-Title'].isin(famous_books)]
famous_books = famous_books.copy()
famous_books.drop_duplicates(subset=['Book-Title'], inplace=True, keep='first')

pt = filtered_books.pivot_table(index='Book-Title',columns='User-ID', values='Book-Rating')
pt.fillna(0, inplace=True)

similarities = cosine_similarity(pt)

def recommend(book_name):
    if book_name in pt.index:
        index = np.where(pt.index == book_name)[0][0]
        similar_books_list = sorted(
        list(enumerate(similarities[index])), key=lambda x: x[1], reverse=True)[1:11]
        
        print(f'Recommendations for the book {book_name}:')
        print('-'*5)
        for book in similar_books_list:
            print(pt.index[book[0]])
        print('\n')

    else:
        print('Book Not Found')
        print('\n')

def recommendUpdated(book_name):
    if book_name in pt.index:
        # Eğer kitap bulunursa önerileri listele
        index = np.where(pt.index == book_name)[0][0]
        similar_books_list = sorted(
            list(enumerate(similarities[index])), key=lambda x: x[1], reverse=True)[1:11]
        
        print(f'Recommendations for the book "{book_name}":')
        print('-'*5)
        for book in similar_books_list:
            print(pt.index[book[0]])
        print('\n')
    else:
        # Eğer kitap yoksa en benzer kitabı bul
        print(f"Book '{book_name}' not found. Finding the most similar book...\n")
        closest_book = process.extractOne(book_name, pt.index)
        
        if closest_book and closest_book[1] > 50:  # Eşik değer olarak %50 benzerlik
            print(f'Most similar book found: "{closest_book[0]}". Showing recommendations...\n')
            recommend(closest_book[0])  # Recursive call for the most similar book
        else:
            print("No similar books found.\n")


recommend('Harry Potter and the Chamber of Secrets (Book 2)')
recommend('Harry Potter and the Chamber of Secrets')
recommendUpdated('Harry Potter and the Chamber of Secrets')
