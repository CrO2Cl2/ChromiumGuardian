import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Read in the data
df = pd.read_csv('spam_data.csv', error_bad_lines=False, encoding='utf-8')

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
print("Training model...")
for i in range(1, X_train.shape[0]):
    clf.partial_fit(X_train[:i], y_train[:i], classes=['spam', 'not spam'])
    print(f"Training progress: {i/X_train.shape[0]*100:.2f}%")

# Save the model and the fitted CountVectorizer object to a file
with open('model.pkl', 'wb') as f:
    pickle.dump((clf, vectorizer), f)

# Evaluate the model on the test set
y_pred = clf.predict(X_test)

# Print the evaluation metrics  the other metrics still don't work
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
#print("Precision:", metrics.precision_score(y_test, y_pred))
#print("Recall:", metrics.recall_score(y_test, y_pred))
#print("F1 score:", metrics.f1_score(y_test, y_pred))

# Compute the confusion matrix
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(confusion_matrix)

# Compute the classification report
classification_report = metrics.classification_report(y_test, y_pred)
print("Classification report:")
print(classification_report)

le = LabelEncoder()

# Encode the labels
y_pred_encoded = le.fit_transform(y_pred)
y_test_encoded = le.transform(y_test)

# Compute the AUC-ROC curve
fpr, tpr, thresholds = metrics.roc_curve(y_test_encoded, y_pred_encoded, pos_label=1)
auc = metrics.roc_auc_score(y_test_encoded, y_pred_encoded)

# Plot the AUC-ROC curve
plt.plot(fpr, tpr, label='AUC-ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()