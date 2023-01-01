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

# Define the message to classify
message = input("message:") # INPUT THE MESSAGES TO CLASSIFY HERE. the input is only for testing purposes

start_time = time.perf_counter()

# Classify the message
prediction = classify_message(message)

end_time = time.perf_counter()

# Print the prediction and time needed for the operation
print(f"Prediction: {prediction}")
print(f"Time needed for the operation: {end_time - start_time:.4f} seconds")
