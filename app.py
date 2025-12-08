import streamlit as st
import pickle
import re

# ==========================
# LOAD MODEL & TF-IDF
# ==========================
model = pickle.load(open("model_sentiment.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# ==========================
# SIMPLE CLEAN TEXT (tanpa NLTK)
# ==========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

# ==========================
# STREAMLIT UI
# ==========================
st.set_page_config(page_title="Amazon Sentiment Analysis", layout="centered")

st.title("Amazon Review Sentiment Analysis")
st.write("Masukkan review produk Amazon untuk mengetahui sentimennya.")

review_input = st.text_area("Masukkan Review:")

if st.button("Prediksi"):
    if review_input.strip() == "":
        st.warning("Masukkan teks review terlebih dahulu.")
    else:
        cleaned = clean_text(review_input)
        vector = tfidf.transform([cleaned])
        prediction = model.predict(vector)[0]

        if prediction == "positive":
            st.success("Sentimen: **POSITIF** 👍")
        elif prediction == "negative":
            st.error("Sentimen: **NEGATIF** 👎")
        else:
            st.info("Sentimen: **NETRAL** 😐")
