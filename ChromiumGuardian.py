#the actual prediction making script

import pickle
import time


# Load the trained model
clf = pickle.load(open('keysmash_model.pkl','rb'))


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
  elif message.startswith("/"):
    return "slash_command"
  else:
    # Use the model to classify new inputs
    prediction = clf.predict([message])
    if prediction[0] == 1:
        print("keysmash")
    else:
        print("message")
