import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import re

# Preprocessing numeric and categorical features
def preprocess_data(df):
    # Clean the price column to extract numeric values
    def clean_price(value):
        try:
            # Extract numeric value using regex
            return float(re.findall(r"[\d.]+", value)[0])
        except (IndexError, ValueError, TypeError):
            # Return 0 if no numeric value is found
            return 0

    df['price'] = df['price'].apply(clean_price)

    # Normalize numeric columns
    scaler = MinMaxScaler()
    df['price'] = scaler.fit_transform(df[['price']])
    
    return df

# Generate weighted TF-IDF vectors for text fields
def create_feature_matrix(df):
    # Initialize TF-IDF Vectorizer for each text field
    tfidf_title = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_description = TfidfVectorizer(stop_words='english', max_features=2000)
    tfidf_category = TfidfVectorizer(stop_words='english', max_features=500)

    # Fit and transform each field
    title_matrix = tfidf_title.fit_transform(df['title'].fillna('')).toarray()
    description_matrix = tfidf_description.fit_transform(df['description'].fillna('')).toarray()
    category_matrix = tfidf_category.fit_transform(df['category'].fillna('')).toarray()

    # Apply weights to each feature
    weighted_title = title_matrix * 0.5  # Lower weight
    weighted_description = description_matrix * 1.0  # Medium weight
    weighted_category = category_matrix * 1.2  # Higher weight

    # Combine all features
    combined_text_features = np.hstack([weighted_title, weighted_description, weighted_category])

    # Numeric features (e.g., price)
    numeric_features = df[['price']].values

    # Combine numeric and text features
    feature_matrix = np.hstack([numeric_features, combined_text_features])
    return feature_matrix

# Function to recommend events for a user
def recommend_events(user_clicked_ids, event_data, similarity_matrix, top_n=5):
    # Get indices of events clicked by the user
    clicked_indices = [event_data[event_data['_id'] == event_id].index[0] for event_id in user_clicked_ids]

    # Compute average similarity scores for each event
    avg_similarity = np.mean(similarity_matrix[clicked_indices], axis=0)

    # Get top N events, excluding already clicked ones
    recommended_indices = np.argsort(-avg_similarity)
    recommended_indices = [idx for idx in recommended_indices if idx not in clicked_indices]

    return event_data.iloc[recommended_indices[:top_n]]

def main():
    # Load the event data
    event_data = pd.read_excel("eventbrite_data_new2.xlsx")  # Replace with your CSV file path

    # Preprocess the data
    event_data = preprocess_data(event_data)

    # Create the feature matrix
    feature_matrix = create_feature_matrix(event_data)

    # Compute pairwise event similarity
    event_similarity = cosine_similarity(feature_matrix)

    # Example user data
    user_clicked_event_ids = [183]  # Replace with actual clicked event IDs

    # Get recommendations
    recommendations = recommend_events(user_clicked_event_ids, event_data, event_similarity, top_n=5)

    # Display recommendations
    print("Recommended Events:")
    print(recommendations[['title', 'description', 'category', 'price']])

if __name__ == "__main__":
    main()
