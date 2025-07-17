# app.py
import pickle
import json
import random
import streamlit as st
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# Load model & metadata
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

with open('intents.json') as f:
    intents = json.load(f)

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

def clean(sentence):
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    lemmatized = [
        lemmatiser.lemmatize(w.lower(), get_word(pos))
        for w, pos in tagged
    ]
    return " ".join(lemmatized)

def get_response(user_input):
    cleaned = clean(user_input)
    vector = vectorizer.transform([cleaned]).toarray()
    predicted = model.predict(vector)[0]
    tag = classes[predicted]
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Sorry, I don't understand that."

# Streamlit UI
st.title("🤖 My Chatbot")
st.markdown("Ask me anything from my knowledge base!")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("You: ", key="input")

if user_input:
    if user_input.lower() in ["quit", "exit"]:
        st.write("👋 Goodbye!")
    else:
        response = get_response(user_input)
        st.session_state.history.append(("You", user_input))
        st.session_state.history.append(("Bot", response))

for speaker, text in reversed(st.session_state.history):
    if speaker == "You":
        st.markdown(f"**🧑 You:** {text}")
    else:
        st.markdown(f"**🤖 Bot:** {text}")
