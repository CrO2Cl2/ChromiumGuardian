import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Read in the data
df = pd.read_csv('spam_data.csv', error_bad_lines=False)

# Create a CountVectorizer object
vectorizer = CountVectorizer()

# Transform the texts into a numerical representation
X = vectorizer.fit_transform(df['text'])

# Extract the labels
y = df['label']

# Split the data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a MultinomialNB object
clf = MultinomialNB()

# Fit the model to the training data
clf.fit(X_train, y_train)

# Define the message to classify
message = "guhvdshjgkdsf"                                                       # INPUT THE MESSAGE TO CLASSIFY HERE

# Transform the message into a numerical representation
X_message = vectorizer.transform([message])
    
# Make a prediction on the message
prediction = clf.predict(X_message)[0]

# Print the prediction
print(f"Prediction: {prediction}")

X_message = vectorizer.transform([message])
    
# Make a prediction on the message
prediction = clf.predict(X_message)[0]

# Print the prediction
print(f"Prediction: {prediction}")
