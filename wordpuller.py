import requests

# Get the text from the Github raw link
url = "https://raw.githubusercontent.com/dwyl/english-words/master/words.txt"
response = requests.get(url)
input_words = response.text.split()
input_words = set(input_words)

#Read words from words.txt file
with open('words.txt', 'r', encoding="utf-8") as f:
    existing_words = f.read().split()
    existing_words = set(existing_words)

#Get the set union of input words and existing words
all_words = input_words | existing_words

#write all words to words.txt file
with open('words.txt', 'a', encoding="utf-8") as f:
    for word in all_words:
        f.write(word + '\n')