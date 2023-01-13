import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from concurrent.futures import ThreadPoolExecutor
import pickle
import time
save_interval = 1000

# Load the valid words from a txt file
with open('words.txt', 'r',  encoding="utf-8") as f:
    valid_words = f.read().splitlines()

# Create a dataframe with the valid words
df = pd.DataFrame(valid_words, columns=["word"])

# Preprocess the data
df['word'] = df['word'].str.replace(r'[^a-zA-Z]', '') # remove non-alphabetic characters
df['word'] = df['word'].str.lower() # convert to lowercase
df = df[df['word'].str.len() > 1] # remove any words shorter than 2 letters

# Create an instance of the CountVectorizer class
vectorizer = CountVectorizer()

# Fit the vectorizer on the word column
vectorizer.fit(df['word'])

# Transform the word column to a numerical representation
X = vectorizer.transform(df['word'])

#saved the vectorizer
with open("vectorizer.pickle", "wb") as f:
    pickle.dump(vectorizer, f)

# Add a column for the label (0 for valid words, 1 for keysmash)
df['is_keysmash'] = 0

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, df['is_keysmash'], test_size=0.2)


# Load the testing set from a csv file
test_df = pd.read_csv('spam_data.csv')
test_df['word'] = test_df['word'].str.replace(r'[^a-zA-Z]', '') # remove non-alphabetic characters
test_df['word'] = test_df['word'].str.lower() # convert to lowercase
test_df = test_df[test_df['word'].str.len() > 3] # remove any words shorter than 4 letters
test_df['is_keysmash'] = test_df['is_spam'].apply(lambda x: 1 if x == 'spam' else 0)
X_test, y_test = test_df['word'], test_df['is_keysmash']

# Create and fit the deep learning model
clf = MLPClassifier(hidden_layer_sizes=(50,50,50), max_iter=1000)

# Set the interval at which to save the model's state
save_interval = 100

# Set the filename to save the model's state
model_filename = 'model.pickle'

# Load the model's state if it exists
try:
    with open(model_filename, 'rb') as f:
        clf = pickle.load(f)
    print("Loaded existing model state.")
except FileNotFoundError:
    print("Starting new model.")

# Train the model
iterations = 0
while iterations < 10000:
    start_time = time.perf_counter()
    with ThreadPoolExecutor() as executor:
        executor.map(clf.fit, X_train, y_train)
    end_time = time.perf_counter()
    tot_time = end_time - start_time
    iterations += 1
    print(iterations)
    print(tot_time)
    if iterations % save_interval == 0:
        with open(model_filename, 'wb') as f:
            pickle.dump(clf, f)
        print("Saved model state at iteration", iterations)

y_pred = clf.predict(X_test)


# Evaluate the model's performance
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", recall)
print("ROC AUC:", roc_auc)

# Plot the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
