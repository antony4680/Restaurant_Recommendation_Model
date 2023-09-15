import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


def Get_Recommendations(customer_id, customer_ratings):
    # Load the data from the pickle file
    with open('rating_full_matrix.pkl', 'rb') as f:
        rating_full_matrix = pickle.load(f)

    # Load customer_similarity from the pickle file
    with open('customer_similarity.pkl', 'rb') as f:
        customer_similarity = pickle.load(f)

    # Load valid_rating_mean from the pickle file
    with open('valid_rating_mean.pkl', 'rb') as f:
        valid_rating_mean = pickle.load(f)

    # Load df from the pickle file
    with open('df.pkl', 'rb') as f:
        df = pickle.load(f)
    
    if customer_id not in rating_full_matrix.index:
        return [85, 676, 195, 92, 106]

    # Vendors which rated by inputted customer
    customer_vendor = rating_full_matrix.loc[customer_id].copy()

    for vendor, rating in customer_ratings.items():
        # Exception for vendors already rated by the customer
        if pd.notnull(customer_vendor.loc[vendor]):
            customer_vendor.loc[vendor] = 0
        else:
            customer_vendor.loc[vendor] = rating

    def cf_knn(customer_id, vendor_id):
        neighbor_size = 0
        if vendor_id in rating_full_matrix:
            # Similarity of inputted customer and other customer
            sim_scores = customer_similarity[customer_id].copy()
            # Ratings by all customers for inputted vendor(restaurant)
            vendor_ratings = rating_full_matrix[vendor_id].copy()
            # Index of customers who are not rate inputted vendor
            none_rating_idx = vendor_ratings[vendor_ratings.isnull()].index
            # Exception rating(null) which of customers who are not rate inputted vendor
            vendor_ratings = vendor_ratings.drop(none_rating_idx)
            # Exception similarity which of customers who are not rate inputted vendor
            sim_scores = sim_scores.drop(none_rating_idx)

            # Case that neighbor size is not specified
            if neighbor_size == 0:
                # Weighted mean of ratings by customers who rate inputted vendor
                mean_rating = np.dot(sim_scores, vendor_ratings) / sim_scores.sum()

            # Case that neighbor size is specified
            else:
                # Case that 2 or more people rate inputted vendor
                if len(sim_scores) > 1:
                    # Minimum value among inputted neighbor size and number of customers who rate inputted vendor
                    neighbor_size = min(neighbor_size, len(sim_scores))
                    # transpose to Numpy array for using argsort
                    sim_scores = np.array(sim_scores)
                    vendor_ratings = np.array(vendor_ratings)
                    # Sorting similarity
                    customer_idx = np.argsort(sim_scores)
                    # Similarity as much as neighbor size
                    sim_scores = sim_scores[customer_idx][-neighbor_size:]
                    # Ratings as much as neighbor size
                    vendor_ratings = vendor_ratings[customer_idx][-neighbor_size:]
                    # Calculate final predicted rating
                    mean_rating = np.dot(sim_scores, vendor_ratings) / sim_scores.sum()
                else:
                    # Substitute to valid mean in other case
                    mean_rating = valid_rating_mean
        else:
            # Substitute to valid mean in other case
            mean_rating = valid_rating_mean
        return mean_rating

    for vendor in rating_full_matrix:
        # Calculate predicted rating about vendors not rated by the customer
        if pd.isnull(customer_vendor.loc[vendor]):
            customer_vendor.loc[vendor] = cf_knn(customer_id, vendor)

    # Sort vendors by predicted rating
    vendor_sort = customer_vendor.sort_values(ascending=False)

    recommended_vendors = []
    for vendor in vendor_sort.index:
        if len(recommended_vendors) == 5:
            break
        if vendor not in recommended_vendors and df.loc[vendor, 'vendor_id'] not in recommended_vendors:
            recommended_vendors.append(df.loc[vendor, 'vendor_id'])

    # Retrieve food IDs from the recommended vendors
    recommended_foods = df.loc[recommended_vendors, 'vendor_id'].tolist()
    return recommended_foods

