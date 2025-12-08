import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# Load Model & TF-IDF
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))


# Cleaning function
def clean_text(t):
    t = t.lower()
    t = re.sub(r'[^a-zA-Z ]','', t)
    t = " ".join([w for w in t.split() if w not in stop_words])
    return t


# UI
st.title("Amazon Review Sentiment Analysis")
st.write("Masukkan review produk Amazon untuk mengetahui sentimennya.")

review = st.text_area("Masukkan Review:")

if st.button("Prediksi"):

    if review.strip() == "":
        st.error("Review tidak boleh kosong!")
    else:
        cleaned = clean_text(review)
        vector = tfidf.transform([cleaned])    # WAJIB!
        prediction = model.predict(vector)[0]

        st.success(f"Hasil Sentiment: **{prediction.upper()}**")
