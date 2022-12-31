import re
import pickle

def classify_message(message):
  # Use regular expressions to check if the message is a Discord invite URL, Tenor link, or HTTPS link
  invitematch = re.search(r"https://(?:www\.)?discord\.gg/\S+", message)
  gifmatch = re.search(r"https://tenor\.com/view/\S+", message)
  httpsmatch = re.search(r"https://\S+", message)
  httpmatch = re.search(r"http://\S+", message)

  if invitematch:
    # Return "discord_invite" if the message is a Discord invite URL
    return "discord_invite"
  elif gifmatch:
    # Return "tenor_gif" if the message is a Tenor link
    return "tenor_gif"
  elif httpsmatch:
    # Return "https_url" if the message is an HTTPS link
    return "https_url"
  elif httpmatch:
    return "http_url"
  else:
    # Load the trained model and the fitted CountVectorizer object from the file
    with open('model.pkl', 'rb') as f:
      clf, vectorizer = pickle.load(f)

    # Transform the message into a numerical representation
    X_message = vectorizer.transform([message])

    # Make a prediction on the message
    prediction = clf.predict(X_message)[0]

    return prediction

# Define the message to classify
message = input("message:") # INPUT THE MESSAGES TO CLASSIFY HERE. the input is only for testing purposes

# Classify the message
prediction = classify_message(message)

# Print the prediction
print(f"Prediction: {prediction}")
