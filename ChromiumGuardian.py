#the actual prediction making script

import pickle
import time

# Load the trained model and the fitted CountVectorizer object from the file
with open('model.pkl', 'rb') as f:
  clf, vectorizer = pickle.load(f)

def classify_message(message):
  # Check if the message is a Discord invite URL, Tenor link, or HTTPS link using string functions
  if message.startswith("https://www.discord.gg/"):
    return "discord_invite"
  elif message.startswith("https://tenor.com/view/"):
    return "tenor_gif"
  elif message.startswith("https://"):
    return "https_url"
  elif message.startswith("http://"):
    return "http_url"
  else:
    # Transform the message into a numerical representation
    X_message = vectorizer.transform([message])

    # Make a prediction on the message
    prediction = clf.predict(X_message)[0]

    return prediction
