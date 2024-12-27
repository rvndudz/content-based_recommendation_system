import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessor
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Preprocesses input text by lowercasing, removing stop words, and lemmatizing.
    """
    tokens = text.lower().split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def compute_similarity(events, user_clicked_event_ids):
    """
    Computes similarity scores for events based on user interactions.
    
    Args:
        events (DataFrame): Event dataset.
        user_clicked_event_ids (list): List of event IDs the user has interacted with.
    
    Returns:
        DataFrame: Events with similarity scores.
    """
    # Combine relevant text fields for TF-IDF
    events['text'] = events['title'] + " " + events['description'] + " " + events['category']
    events['text'] = events['text'].apply(preprocess_text)
    
    # Create TF-IDF matrix
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(events['text'])
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Get indices for clicked events
    clicked_indices = [events[events["_id"] == event_id].index[0] for event_id in user_clicked_event_ids]
    
    # Calculate average similarity for clicked events
    similarity_scores = cosine_sim[clicked_indices].mean(axis=0)
    
    # Add similarity scores to the events DataFrame
    events['similarity_score'] = similarity_scores
    
    # Exclude already clicked events
    recommendations = events[~events["_id"].isin(user_clicked_event_ids)].sort_values(by='similarity_score', ascending=False)
    
    return recommendations

# Main Function
if __name__ == "__main__":
    # Load events dataset (from CSV or database)
    # Example: events = pd.read_csv('events.csv')
    events = pd.read_csv("events.csv")  # Replace with your actual dataset
    events['_id'] = events['_id'].astype(str)  # Ensure IDs are strings
    
    # Example user clicked event IDs
    user_clicked_event_ids = ["1", "3"]  # Replace with actual user interaction data
    
    # Compute recommendations
    recommendations = compute_similarity(events, user_clicked_event_ids)
    
    # Display top recommendations
    print(recommendations[['title', 'similarity_score']])
