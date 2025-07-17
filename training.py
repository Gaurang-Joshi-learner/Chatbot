# train.py
import nltk
import pickle
import json
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

lemmatiser = WordNetLemmatizer()

def get_word(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Load intents
with open('intents.json') as f:
    intents = json.load(f)

documents = []
labels = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        tokens = word_tokenize(pattern)
        tagged = pos_tag(tokens)
        lemmatized = [
            lemmatiser.lemmatize(w.lower(), get_word(pos)) 
            for w, pos in tagged
        ]
        processed = " ".join(lemmatized)
        documents.append(processed)
        labels.append(intent['tag'])

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents).toarray()

classes = list(set(labels))
y = [classes.index(tag) for tag in labels]

model = MultinomialNB()
model.fit(X, y)

pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

print("✅ Model trained and saved!")
