# Train the model and save it to a file
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Read in the data
df = pd.read_csv('spam_data.csv', error_bad_lines=False, encoding='windows-1252')

# Create a CountVectorizer object
vectorizer = CountVectorizer()

# Transform the texts into a numerical representation
X = vectorizer.fit_transform(df['text'])

# Extract the labels
y = df['label'].values

# Split the data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a MultinomialNB object
clf = MultinomialNB()

# Fit the model to the training data
clf.fit(X_train, y_train)

# Save the model and the fitted CountVectorizer object to a file
with open('model.pkl', 'wb') as f:
    pickle.dump((clf, vectorizer), f)
