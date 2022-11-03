import streamlit as st
import pickle
import numpy as np
import pyidaungsu as pds
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from keras.models import load_model

def tokenize(line):
    sentence = pds.tokenize(line,form="word")
    return sentence

loaded_model = load_model('model_new.h5')
vectorizer = pickle.load(open("vectorizer_new.pickle", "rb"))

stopwordslist = []
slist = []
with open("./stopword.txt", encoding = 'utf8') as stopwordsfile:
    stopwords = stopwordsfile.readlines()
    slist.extend(stopwords)
    for w in range(len(slist)):
        temp = slist[w]
        stopwordslist.append(temp.rstrip())

def stop_word(sentence):
  new_sentence = []
  for word in sentence.split():
    if word not in stopwordslist:
      new_sentence.append(word)
  return(' '.join(new_sentence))

def tokenize(line):
    sentence = pds.tokenize(line,form="word")
    sentence = ' '.join([str(elem) for elem in sentence])
    sentence = stop_word(sentence)
    return sentence

st.title('Automatic Sentiment Analysis for Myanmar Language')
st.subheader("ERS NLP")
sentence = st.text_area("Enter your news Content Here", height=200)
sentence = tokenize(sentence)
predict_btt = st.button("Predict")
if predict_btt:
  data = vectorizer.transform([sentence]).toarray()
  prediction = loaded_model.predict(data)
  if prediction>0.9900:
    st.text("Positive")
    print(prediction)
  elif 0.0500<prediction<0.9900:
    st.text("Natural")
    print(prediction)
  elif 0.0005<prediction<0.0500:
    st.text("Negative")
    print(prediction)
  elif 0.0000<prediction<0.0005:
    st.text("Strong Negative")
    print(prediction)
  
  

