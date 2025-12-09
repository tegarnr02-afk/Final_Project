import streamlit as st
import pickle
import re
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load stopwords
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

# Load Model & TF-IDF
model = pickle.load(open("model_sentiment.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# Cleaning text function
def clean_text(t):
    t = t.lower()
    t = re.sub(r'[^a-zA-Z ]', '', t)
    t = " ".join([w for w in t.split() if w not in STOPWORDS])
    return t

# Streamlit Page Config
st.set_page_config(
    page_title="Amazon Sentiment Analyzer",
    page_icon="🛒",
    layout="centered",
)

# ---------- HEADER ----------
st.markdown("""
    <h1 style="text-align:center; color:#4b7bec;">
        🛒 Amazon Review Sentiment Analyzer
    </h1>
    <p style="text-align:center; font-size:18px;">
        Masukkan review produk — aplikasi akan memprediksi apakah review tersebut 
        <b style='color:green;'>Positive</b>, 
        <b style='color:orange;'>Neutral</b>, atau 
        <b style='color:red;'>Negative</b>.
    </p>
""", unsafe_allow_html=True)

st.write("---")

# ---------- INPUT ----------
review_input = st.text_area("✍️ Tulis review produk di sini:", height=150)

if st.button("🔍 Prediksi Sentimen"):
    if review_input.strip() == "":
        st.warning("Masukkan teks review terlebih dahulu.")
    else:
        clean = clean_text(review_input)
        vector = tfidf.transform([clean])

        pred = model.predict(vector)[0]
        proba = model.predict_proba(vector).max()

        # Warna label
        if pred == "positive":
            color = "#2ecc71"
        elif pred == "negative":
            color = "#e74c3c"
        else:
            color = "#f1c40f"

        st.markdown(f"""
            <div style="padding:18px; border-radius:10px; background-color:{color}; color:white;">
                <h3 style="text-align:center;">🎯 Hasil Prediksi</h3>
                <h2 style="text-align:center;">{pred.upper()}</h2>
                <p style="text-align:center;">Confidence Score: <b>{proba:.2f}</b></p>
            </div>
        """, unsafe_allow_html=True)

st.write("---")

# ---------- FEATURES ----------
st.subheader("📊 Wordcloud Contoh Kata Review")
sample_text = "product good quality fast charging broke damage amazing bad waste money return"
wc = WordCloud(width=700, height=350, background_color="white").generate(sample_text)

plt.figure(figsize=(10,5))
plt.imshow(wc)
plt.axis("off")
st.pyplot(plt)
