import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

cus_ven_ratings = pd.read_csv('new_cus_ven_ratings.csv')

ratings_not_none = []

for i in range(0, cus_ven_ratings.shape[0]-1) :
  if pd.isnull(cus_ven_ratings.iloc[i][2]) == False and cus_ven_ratings.iloc[i][2] != 0 :
    ratings_not_none.append(cus_ven_ratings.iloc[i][2])
    
valid_rating_mean = np.mean(np.array(ratings_not_none))

def rating_missing_func(x) :
  if pd.isnull(x) == True :
    return valid_rating_mean
  elif x == 0 :
    return valid_rating_mean
  else :
    return x

cus_ven_ratings["rating2"] = cus_ven_ratings["rating"].apply(rating_missing_func)

cus_ven_ratings = cus_ven_ratings[['customer_id', 'vendor_id', 'rating2']]
cus_ven_ratings.rename(columns={'rating2':'rating', 1:'customer_id_num'}, inplace=True)


cus_ven_ratings_mean = cus_ven_ratings.groupby(['customer_id', 'vendor_id']).mean()

df_cus_ven_ratings_mean = cus_ven_ratings_mean.reset_index()

rating_full_matrix = df_cus_ven_ratings_mean.pivot(index='customer_id', columns='vendor_id', values='rating')

from sklearn.metrics.pairwise import cosine_similarity
rating_matrix_dummy = rating_full_matrix.copy().fillna(0)

customer_similarity = cosine_similarity(rating_matrix_dummy, rating_matrix_dummy)

customer_similarity = pd.DataFrame(customer_similarity, index = rating_full_matrix.index, columns=rating_full_matrix.index)

df = pd.read_csv('df1_train_for_anal.csv')

import pickle

# Pickle rating_full_matrix
with open('rating_full_matrix.pkl', 'wb') as f:
    pickle.dump(rating_full_matrix, f)

# Pickle customer_similarity
with open('customer_similarity.pkl', 'wb') as f:
    pickle.dump(customer_similarity, f)

# Pickle valid_rating_mean
with open('valid_rating_mean.pkl', 'wb') as f:
    pickle.dump(valid_rating_mean, f)

# Pickle df
with open('df.pkl', 'wb') as f:
    pickle.dump(df, f)