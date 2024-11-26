#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import seaborn as sns
import numpy as np
import warnings
import matplotlib.pyplot as plt
import os
import nltk


# # Importing Datasets
# The datasets used in this file were downloaded from kaggle
# > Link: https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset
# 
# There are three datasets available for us:
# 1. Books
# 2. Users
# 3. Ratings

books = pd.read_csv('datasets/Books.csv')
ratings = pd.read_csv('datasets/Ratings.csv')
users = pd.read_csv('datasets/Users.csv')


# # Shape of Data
# We can see that we have more ratings than the number of books.

books.isna().sum()


# Lets fix the author of the missing book and try to find the images url too.

books[books['Book-Author'].isna()]


# # Filling missing values of Books datasets

books.iloc[187689]['Book-Title']


# After some google searches, we find that the author of this book is:<br>
# ***Downes, Larissa Anne***

books.loc[187689, 'Book-Author'] = 'Downes, Larissa Anne'


books.isna().sum()


# Users dataset
users.isna().sum()


# # Dropping Age Column
# As we can see, there are lots of missing values in age, it is ideal to drop the columns since imputation of these many users will cause the model to be biased and incorrect.

users.drop(columns=['Age'], inplace=True)


ratings.isnull().sum()


# Great, there are no missing values. So we dont have to drop anything.

# # Checking duplicated values
# Lets check for duplicated values in all datasets and drop them.

# Books dataset
books.duplicated().sum()


# Users dataset
users.duplicated().sum()


# Ratings dataset
ratings.duplicated().sum()


# There are no duplicate values in the datasets.

# # EDA
# Lets perform EDA on the data

# ## Books Dataset

books.head()


books.dtypes


# # Fixing Data
# There has been some error in the year of publication on the index of:<br>
# **209538**
# <br>
# **220731**
# <br>
# **221678**
# <br>
# So, we fix those data entries
# 

books.iloc[209538]


books.iloc[209538]['Year-Of-Publication']


books.loc[209538, 'Book-Author'] = 'Michael Teitelbbaum'
books.loc[209538, 'Book-Title'] = 'DK Readers: The Story of the X-Men, How It All Began (Level 4: Proficient Readers)'
books.loc[209538, 'Year-Of-Publication'] = 2000
books.loc[209538, 'Publisher'] = 'DK Publishing Inc'


books.iloc[220731]


books.loc[220731, 'Book-Title'] = "Peuple du ciel, suivi de 'Les Bergers'"
books.loc[220731, 'Book-Author'] = 'Jean-Marie Gustave Le Clézio'
books.loc[220731, 'Year-Of-Publication'] = 1990
books.loc[220731, 'Publisher'] = 'Gallimard'


books.iloc[221678]


books.iloc[221678]['Book-Title']


books.loc[221678, 'Book-Title'] = 'DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)'
books.loc[221678, 'Book-Author'] = 'James Buckley'
books.loc[221678, 'Year-Of-Publication'] = 2000
books.loc[221678, 'Publisher'] = 'DK Publishing Inc'


books['Year-Of-Publication'] = books['Year-Of-Publication'].astype('int64')


books['Year-Of-Publication'].value_counts().sort_index(ascending=False).iloc[:20]


# # Mistakes in Data?
# We can see that there are few data with year of publication from future.

books[books['Year-Of-Publication']>2021][['Book-Title','Year-Of-Publication','Publisher','Book-Author']]





# # Fixing the year again
# We can see the above books have years with publishing date from future, which is not possible.
# 

books.loc[37487, 'Year-Of-Publication'] = 1991
books.iloc[37487]


#'MY TEACHER FRIED MY BRAINS (RACK SIZE) (MY TEACHER BOOKS)'
books.loc[37487, 'Year-Of-Publication'] = 1991

# 'MY TEACHER FLUNKED THE PLANET (RACK SIZE) (MY TEACHER BOOKS)'
books.loc[55676, 'Year-Of-Publication'] = 2005

# 'MY TEACHER FLUNKED THE PLANET (RACK SIZE) (MY TEACHER BOOKS)'
books.loc[37487, 'Book-Author'] = 'Bruce Coville'

# "Alice's Adventures in Wonderland and Through the Looking Glass (Puffin Books)"
books.loc[80264, 'Year-Of-Publication'] = 2003

# 'Field Guide to the Birds of North America, 3rd Ed.'
books.loc[192993, 'Year-Of-Publication'] = 2003

# Crossing America
books.loc[78168, 'Year-Of-Publication'] = 2001

# Outline of European Architecture (Pelican S.)
books.loc[97826, 'Year-Of-Publication'] = 1981

# Three Plays of Eugene Oneill
books.loc[116053, 'Year-Of-Publication'] = 1995

# Setting to current date of project since no information could be found
# Das groÃ?Â?e BÃ?Â¶se- MÃ?Â¤dchen- Lesebuch.
books.loc[118294, 'Year-Of-Publication'] = 2023

# FOREST PEOPLE (Touchstone Books (Hardcover))
books.loc[228173, 'Year-Of-Publication'] = 1987

# In Our Time: Stories (Scribner Classic)
books.loc[240169, 'Year-Of-Publication'] = 1996

# CLOUT
books.loc[246842, 'Year-Of-Publication'] = 1925

# To Have and Have Not
books.loc[255409, 'Year-Of-Publication'] = 1937

# FOOTBALL SUPER TEAMS : FOOTBALL SUPER TEAMS
books.loc[260974, 'Year-Of-Publication'] = 1991


books['Year-Of-Publication'].value_counts().sort_index(ascending=False).iloc[:20]


# Looks like we have a lot of books without the year data and hence set to 0

books[(books['Year-Of-Publication']<1400)&(books['Year-Of-Publication']>0)]


# These are probably mythological and religious books so the date could be correct. So, we donot alter the date for these entries.

books_year_rational = books[books['Year-Of-Publication']!=0]['Year-Of-Publication'].value_counts().sort_index(ascending=False).iloc[:20]
books_year_rational





# ## Authors
# Lets find out if there authors with multiple books or not

books[books['Book-Author'].duplicated()]


# Lets find their number of books per author

# Number of unique authors
len(books['Book-Author'].unique())


author_book_count = books['Book-Author'].value_counts()
author_book_count.head(20)


# ### Error in author name:
# Looks like there are a lot of books with no authors. So lets drop them for a while and count the rest

author_book_count = books[books['Book-Author']!= 'Not Applicable (Na )']
author_book_count_top50 = author_book_count.groupby('Book-Author').count()['Book-Title'].sort_values(ascending=False).head(50)
author_book_count_top50.head(10)


cool = sns.color_palette("cool", n_colors=len(author_book_count_top50.values))



publishers = books['Publisher'].value_counts()
publishers


top_50_publishers = publishers.sort_values(ascending=False)[:51]
top_50_publishers


cool = sns.color_palette("cool", n_colors=len(author_book_count_top50.values))



# # Ratings Dataset
# We have two columns in the ratings dataset with columns: ISBN and rating along with user-ID
# Since, it is difficult to identify books based on ISBN, we merge the datasets **Books** and **Ratings** on the ISBN column

ratings.head()



bookRating = pd.merge(ratings, books, on="ISBN")
bookRating.head()


bookRating.shape


# ## Dropping useless columns
# Since, we donot need the URL here, we can drop the image url columns

bookRating.drop(columns=['Image-URL-S','Image-URL-M','Image-URL-L'],inplace=True)


bookRating.head()


# # Calculating the Average Rating of all Books

averageRating = pd.DataFrame(bookRating.groupby('ISBN')['Book-Rating'].mean().round(1))
averageRating.reset_index(inplace=True)
averageRating.head()


averageRating.shape
averageRating.rename(columns={'Book-Rating':'Average-Rating'}, inplace=True)
averageRating.head()


averageRatingdf = pd.merge(bookRating, averageRating, on='ISBN')
averageRatingdf.head()


averageRatingdf.shape


averageRatingOnly = averageRatingdf[['ISBN','Average-Rating']]
averageRatingOnly.head()


averageRatingUnique = averageRatingOnly[['ISBN','Average-Rating']].drop_duplicates(subset=['ISBN'])
averageRatingUnique.head()


ratingBooks = pd.merge(books, averageRatingUnique, on='ISBN', how='inner')


averageRatingUnique.shape


books_with_rating = pd.merge(books, averageRatingUnique, on='ISBN')
books_with_rating.shape


books_with_rating = books_with_rating[['ISBN','Book-Title','Book-Author','Average-Rating','Year-Of-Publication','Publisher','Image-URL-S','Image-URL-M','Image-URL-L']]
books_with_rating.head()


# # EDA on Average Rating of Book
# Viewing the top 30 books with highest average rating

books_with_rating.sort_values(by=['Average-Rating'], ascending=False).head(30)


ratings_sorted = books_with_rating['Average-Rating'].value_counts().sort_index(ascending=False)
books_with_rating['Average-Rating'].value_counts(normalize=True).round(4).sort_index(ascending=False)




top_20_ratings = books_with_rating['Average-Rating'].value_counts().drop(index=0.0).sort_values(ascending=False).head(20)



# ## Data Visualization results
# We can see the most rated books are **5**,**8**, and **10**

users.head()


len(users.Location.unique())


users.Location.value_counts()


# # MACHINE LEARNING AND BUILDING A MODEL

# We would be making and deploying a ML algorithm for book recommendation system.

# ## Popularity Based Recommender System
# We would select the top 50 books that have the highest average ratings and then display them over to the user. <br> But the problem with this system is that if some book has low number of votes but high rating then it could get biased and proportionally incorrect.<br>So, inorder to avoid that we will set a criteria where we only select the average ratings of books that have been voted by at least **100 users**.

ratings_books_merged = ratings.merge(books, on='ISBN')
ratings_books_merged.shape


# ## Number of Votes
# Lets count the number of votes for each book
# <br>

ratings_books_nonzero = ratings_books_merged[ratings_books_merged['Book-Rating']!=0]
num_rating_df = ratings_books_nonzero.groupby('Book-Title').count()['Book-Rating'].sort_values(ascending=False).reset_index()
num_rating_df.rename(columns={'Book-Rating':'Number-of-Ratings'}, inplace=True)


# # Average Book Rating
# Lets find the average ratings of books

avg_rating_df = ratings_books_nonzero.groupby('Book-Title').mean(numeric_only=True)['Book-Rating'].reset_index()
avg_rating_df.rename(columns={'Book-Rating':'Average-Rating'}, inplace=True)
avg_rating_df.head()


# Merging these two dataframes

popularity_df = pd.merge(num_rating_df, avg_rating_df, on='Book-Title')
popularity_df


# Taking the books whose number of ratings are greater than 100

popularity_df_above_100 = popularity_df[popularity_df['Number-of-Ratings']>=100]
popularity_df_above_50 = popularity_df[popularity_df['Number-of-Ratings'] >= 50]
popularity_df_above_250 = popularity_df[popularity_df['Number-of-Ratings'] >= 250]
popularity_df_above_100.sort_values(by='Average-Rating', ascending=False).head()


# # New metric for rating calculation
# We still have a problem if there are low number of high ratings, and high number of lower ratings, we get a bias and unfairness in the rating scenario.
# <br>
# To tackle this, lets use a new metric known as weighted metric which can be calculated by:
# 
# 
# weighted_rating = (average_rating * number_of_ratings + minimum_threshold * default_rating)/(number_of_ratings + minimum_threshold)
# <br><br>
# where,<br><br>
# **average_rating** is available from dataframe<br>
# **number_of_ratings** is available from dataframe<br>
# **minimum_threshold** is the minimum number of votes taken for validation. Here, **100** <br>
# **default_rating** is the neutral state. Here, **5.0** <br>

# Defining a new function that can calculate the metric
def calcWeightedRating(row, avgRating, numOfRatings, minThres, defRating):
    weightedRating = ((row[avgRating] * row[numOfRatings]) + (minThres * defRating))/(row[numOfRatings] + minThres)
    return weightedRating


# Great, lets apply this in our existing dataframe

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


# # Merging with the books dataframe
# We merge the popularity_df_above_250 with books dataframe to get the information about the book author and other details.

popular_df_merge = pd.merge(popularity_df_above_100, books, on='Book-Title').drop_duplicates('Book-Title',keep='first')
popular_df_merge = popular_df_merge.drop(columns=['Image-URL-S', 'Image-URL-L'])
popular_df_merge.sort_values('Weighted-Rating', ascending=False).head(10)


# # TOP RATED BOOKS
# The books shown above are the top rated books and could be shown in the home page of the app

# ## Collaborative Filtering Based Approach
# For user based recommendation system, we build a new type of data like:
# 
# Where each user would have rating for each book and we would also follow criterias like:
# 
# 1. **Value experienced users ratings:**
#     
#     We would take the ratings of the users who have rated over 200 books i.e. have studied and reviewed high amount of books with the highest priority.
#     
# 2. **Value books with high ratings:**
#     
#     We would only recommend books that have more than 50 ratings i.e. popular and famous books.

ratings_books_merged.head()


# ### Filtering Users
# Let's find users who have made at least 200 votes.

users_ratings_count = ratings_books_merged.groupby('User-ID').count()['ISBN']
users_ratings_count = users_ratings_count.sort_values(ascending=False).reset_index()
users_ratings_count.rename(columns={'ISBN':'No-of-Books-Rated'}, inplace=True)
users_ratings_count.head()


# There are 92106 users who have rated the books.

users_200 = users_ratings_count[users_ratings_count['No-of-Books-Rated']>=200]


# There are only 816 users who have rated more than 200 times

books_with_users_200 = pd.merge(users_200, ratings_books_merged, on='User-ID')
books_with_users_200.head()


# There are 475007 books that have been voted by the users who have rated more than 200 times.
# 
# ### Finding books with more than 50 ratings
# Lets find books with more than 50 ratings.

ratings_books_merged.head()


books_ratings_count = ratings_books_merged.groupby('Book-Title').count()['ISBN'].sort_values(ascending=False).reset_index()
books_ratings_count.rename(columns={'ISBN':'Number-of-Book-Ratings'}, inplace=True)
books_ratings_count.head()


books_ratings_50 = books_ratings_count[books_ratings_count['Number-of-Book-Ratings']>=50]

books_ratings_50.head()


# ### Merging the filtered users and books

filtered_books = pd.merge(books_ratings_50, books_with_users_200,  on='Book-Title')
filtered_books.head()


famous_books = filtered_books.groupby('Book-Title').count().reset_index()
famous_books = famous_books['Book-Title']
famous_books = books[books['Book-Title'].isin(famous_books)]
famous_books = famous_books.copy()
famous_books.drop_duplicates(subset=['Book-Title'], inplace=True, keep='first')
famous_books


pt = filtered_books.pivot_table(index='Book-Title',columns='User-ID', values='Book-Rating')
pt.fillna(0, inplace=True)
pt


# # Working of model:
# 
# We make each book as a point made by a vector using the users. That means each books is represented as rating by the user that have made more than 200 votes. Here, 816 users.
# 
# After representing them as cosine vectors we then proceed to use the cosine similarity as the measure of similarity.

from sklearn.metrics.pairwise import cosine_similarity


similarities = cosine_similarity(pt)
similarities


similarities.shape


# # Making a function

# Lets make a function that can recommend books based on the given similarity

# ## Step 1: Retrieving Index

# Retrieving the index of movie
np.where(pt.index=='1984')


# Its in the 3rd place of pivot column

np.where(pt.index=='stardust')[0][0]


# ## Step 2: Use the index to find the array from inside the similarity

# Chaining from the above
similarities[np.where(pt.index=='stardust')[0][0]]




list(enumerate(similarities[0]))


sorted(list(enumerate(similarities[0])), key=lambda x: x[1], reverse=True)


sorted(list(enumerate(similarities[0])), key=lambda x: x[1], reverse=True)[1:6]



for book in sorted(list(enumerate(similarities[0])), key=lambda x: x[1], reverse=True)[1:6]:
    print(book[0])



for book in sorted(list(enumerate(similarities[0])), key=lambda x: x[1], reverse=True)[1:6]:
    print(pt.index[book[0]])


if 'hamkmfa' in pt.index:
    np.where(pt.index=='hamkda')[0][0]
else:
    print('Book not found')

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

recommend('Harry Potter and the Chamber of Secrets (Book 2)')
recommend('1984')
recommend('Message in a Bottle')
recommend('The Da Vinci Code')
recommend('The Return of the King (The Lord of the Rings, Part 3)')
recommend('The Hobbit')
recommend('19')

