import nltk
import streamlit as st
import pickle
import string
nltk.download('punkt')
nltk.download('stopwords')
from sklearn.naive_bayes import MultinomialNB
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

tfidf = pickle.load(open('tfidf.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalpha()]
    text = [word for word in text if word not in nltk.corpus.stopwords.words('english')]
    text = [ps.stem(word) for word in text]
    text = ' '.join(text)
    return text


st.title("SMS Spam Classifier")
input_sms = st.text_input("Enter the Message:")

if st.button('Predict'):

    # preprocess
    transform_sms = transform_text(input_sms)

    # vectorize
    vector_input = tfidf.transform([transform_sms])

    # predict
    result = model.predict(vector_input)[0]

    # display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

