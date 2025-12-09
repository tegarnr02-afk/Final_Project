import streamlit as st
import pickle
import re

# ================================
# 1. Load Model & TFIDF
# ================================
model = pickle.load(open("model_sentiment.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# ================================
# 2. Stopword Manual (tanpa NLTK)
# ================================
STOPWORDS = {
    "the","is","and","to","for","this","that","it","on","in","with","was","at","as",
    "are","be","have","has","had","a","an","of","or","so","but","very","too","not"
}

# ================================
# 3. Clean Text
# ================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    words = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(words)

# ================================
# 4. Predict Function
# ================================
def predict_sentiment(text):
    cleaned = clean_text(text)
    vector = tfidf.transform([cleaned])
    pred = model.predict(vector)[0]
    return pred

# ================================
# 5. UI Design
# ================================
st.set_page_config(page_title="Amazon Review Sentiment", page_icon="⭐")

st.markdown(
    """
    <h2 style="text-align:center; color:#4CAF50;">
        ⭐ Amazon Product Review Sentiment Analyzer ⭐
    </h2>
    <p style="text-align:center; font-size:16px;">
        Masukkan review produk — sistem akan memprediksi apakah sentiment <b>Positive</b>, <b>Neutral</b>, atau <b>Negative</b>.
    </p>
    """,
    unsafe_allow_html=True
)

# ================================
# 6. Input Box
# ================================
user_input = st.text_area("Ketikkan review produk:", height=150)

btn = st.button("🔍 Prediksi Sentiment")

# ================================
# 7. Prediction Output
# ================================
if btn:
    if user_input.strip() == "":
        st.warning("Tolong isi review terlebih dahulu.")
    else:
        result = predict_sentiment(user_input)

        if result == "positive":
            st.success("✅ **Sentiment: Positive** \n🎉 Produk dinilai baik!")
        elif result == "neutral":
            st.info("ℹ️ **Sentiment: Neutral**")
        else:
            st.error("❌ **Sentiment: Negative** \n⚠️ Mungkin produk tidak memuaskan.")

# ================================
# 8. Bonus Fitur
# ================================
st.markdown("### 📊 Fitur Tambahan")
st.caption("Anda bisa menambahkan fitur seperti upload CSV untuk analisis massal, atau rekomendasi produk berdasarkan sentiment.")
