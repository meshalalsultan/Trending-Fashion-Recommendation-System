# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import numpy as np

# Load your dataset
@st.cache_data  # Use the cache to load data only once
def load_data():
    data = pd.read_csv('fashon_data/Myntra kurtis.csv')
    # Perform any preprocessing or feature engineering if not already done
    # (e.g., handling missing values, standardizing brand names, etc.)
    # Replace missing values in 'Product Ratings' with a neutral rating (assuming 2.5 on a scale of 5)
    data['Product Ratings'].fillna(2.5, inplace=True)

# Drop rows where 'Selling Price', 'Price', or 'Discount' is missing
    data.dropna(subset=['Selling Price', 'Price', 'Discount'], inplace=True)

# 1.2 Clean and Standardize Data

# Standardize Brand Names (example: convert to uppercase)
    data['Brand Name'] = data['Brand Name'].str.upper()

# Extract numerical values from 'Discount' column (e.g., '40% OFF' to 40)
    data['Discount'] = data['Discount'].str.extract('(\d+)').astype(float)

# Display the cleaned data
    data.head()
    

# 3.1 Popularity Index
    data['Popularity Index'] = data['Product Ratings'] * np.log(data['Number of ratings'] + 1)

# 3.2 Price Range
    price_bins = [0, 500, 1000, 2000, np.inf]  # Define price bins
    price_labels = ['Low', 'Medium', 'High', 'Premium']
    data['Price Range'] = pd.cut(data['Selling Price'], bins=price_bins, labels=price_labels)

# Display the data with the new features
    data[['Brand Name', 'Product Ratings', 'Number of ratings', 'Popularity Index', 'Selling Price', 'Price Range']].head()

    
    # One-Hot Encoding of 'Price Range'
    one_hot_encoder = OneHotEncoder()
    price_range_encoded = one_hot_encoder.fit_transform(data[['Price Range']]).toarray()

    # Normalize 'Product Ratings' and 'Popularity Index'
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data[['Product Ratings', 'Popularity Index']])

    # Concatenate all features to form the feature vector for each product
    feature_vectors = np.concatenate((data_normalized, price_range_encoded), axis=1)
    
    # Calculate cosine similarity matrix
    cosine_sim_matrix = cosine_similarity(feature_vectors)
    
    return data, cosine_sim_matrix

data, cosine_sim_matrix = load_data()

# Define the recommendation function
def recommend_products_by_id(product_id, number_of_recommendations=5):
    # ... (your existing recommendation logic)
    if product_id >= len(data):
        return f"No product found with ID: {product_id}"
    
    # Get similarity values with other products
    similarity_scores = list(enumerate(cosine_sim_matrix[product_id]))
    
    # Sort the products based on similarity scores
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the top N most similar products (excluding itself)
    similarity_scores = similarity_scores[1:number_of_recommendations+1]
    
    # Get product indices
    product_indices = [i[0] for i in similarity_scores]
    
    # Return the top N most similar products
    return data.iloc[product_indices]

# Streamlit user interface
st.title('Fashion Recommendation System')
st.write('Enter a product ID to get recommendations:')

# User input for product ID
product_id = st.number_input('Product ID', min_value=0, max_value=len(data)-1, value=0, step=1)

if st.button('Show Recommendations'):
    recommendations = recommend_products_by_id(product_id, 5)
    st.write('Recommendations for Product ID {}:'.format(product_id))
    st.dataframe(recommendations[['Brand Name', 'Product Ratings', 'Popularity Index', 'Selling Price', 'Price Range']])
    st.write('Made By Meshal Sultan')
    st.markdown('Check out for more project https://meshalalsultan.com')
    st.markdown('My Linkedin www.linkedin.com/in/meshalhandal')
    st.markdown('The code will be in Github https://github.com/meshalalsultan')

# Run this in your terminal to start the Streamlit app:
# streamlit run your_script_name.py
