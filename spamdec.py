# Classify messages using the trained model
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Define the message to classify
messages = ["uighigh"]                     # INPUT THE MESSAGES TO CLASSIFY HERE

# Load the trained model and the fitted CountVectorizer object from the file
with open('model.pkl', 'rb') as f:
    clf, vectorizer = pickle.load(f)

for message in messages:
    # Transform the message into a numerical representation
    X_message = vectorizer.transform([message])
        
    # Make a prediction on the message
    prediction = clf.predict(X_message)[0]
    
    # Print the prediction
    print(f"Prediction: {prediction}")
