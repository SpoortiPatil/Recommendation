# Importing the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
!pip install mlxtend
from mlxtend.frequent_patterns import apriori, association_rules

# Importing the dataset
book= pd.read_csv("book_id.csv", sep=',', encoding='latin-1')
book.columns
book.describe
book= book.rename(columns= {"Book.Rating" : "rating"})
book.columns

# checking NA values
book.isna().sum()

# Value counts for each rating
value_counts= book.rating.value_counts()

# Barplot of value counts of each rating
plt.bar(
        x= value_counts.index,
        height= value_counts,
        color= "rgbkymc");
plt.xlabel("counts"); plt.ylabel("index")
# The non rating value i.e. 0 has highest counts, ratings 8 and 7 has got the next highest counts
book_rating= book

# Checking the books with highest total rating counts
book_ratingCount= (book_rating.
                   groupby(by=['Book.Title'])['rating'].
                   count().
                   reset_index().
                   rename(columns= {'rating':'totalRatingCount'})
                   [['Book.Title', 'totalRatingCount']]
                   )

book_ratingCount.head()
book_ratingCount.shape
book_ratingCount.sort_values('totalRatingCount', ascending= False, inplace= True)

pd.set_option('display.float_format',lambda x: '%.3f' % x)
book_ratingCount.describe()

# Only few books have more than one rating and remaining all the books have only one rating
rating_with_totalRatingCount = book_ratingCount.merge(book_rating, left_on ='Book.Title',right_on='Book.Title',how='left')
rating_with_totalRatingCount.head()
rating_with_totalRatingCount.shape

# Renaming the column name for convinience
rating_with_totalRatingCount= rating_with_totalRatingCount.rename(columns={'Book.Rating': 'rating'})

# Considering only the books which have rating of minimum 5 
popularity_threshold = 5
rating_popular_book = rating_with_totalRatingCount.query('rating >= @popularity_threshold')
rating_popular_book.head()
rating_popular_book.columns
rating_popular_book.info()

# presenting the dataframe in pivot table form
books_features_df = rating_popular_book.pivot_table(index='Book.Title',columns='User.ID',values='rating').fillna(0)
# Converting the dataframe into rating matrix
from scipy.sparse import csr_matrix
books_features_df_matrix = csr_matrix(books_features_df)

# USING THE KNN FOR COLLBARATIVE FILTERING
from sklearn.neighbors import NearestNeighbors

model_knn = NearestNeighbors(metric="cosine",algorithm="brute")
model_knn.fit(books_features_df_matrix)

NearestNeighbors(algorithm='brute', metric='cosine')
books_features_df.shape

# Randomly selecting any book from the dataframe
query_index = np.random.choice(books_features_df.shape[0])
query_index

# Defining the distances and indices for the KNN model
distances, indices = model_knn.kneighbors(books_features_df.iloc[query_index,:].values.reshape(1,-1),n_neighbors=5)
for i in range(0, len(distances.flatten())):
    if i == 0:
        print("Recommendation for {0}:\n".format(books_features_df.index[query_index]))
    else:
        print('{0}:{1},with distance of {2}'.format(i, books_features_df.index[indices.flatten()[i]],distances.flatten()[i]))

# The above 4 books are good recommendations for the randomly selected book.