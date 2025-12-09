import streamlit as st
import pickle
import re

# ===============================
# 1. Load Model & TF-IDF
# ===============================
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# ===============================
# 2. Text Cleaning (tanpa nltk)
# ===============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ===============================
# 🎨 Custom CSS Aesthetic
# ===============================
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .title {
        font-size: 32px;
        font-weight: 700;
        color: #3A3A3A;
        text-align: center;
        margin-top: -20px;
    }
    .subtitle {
        font-size: 18px;
        color: #555;
        text-align: center;
        margin-bottom: 20px;
    }
    .result-box {
        padding: 15px;
        border-radius: 12px;
        background-color: #ffffff;
        border: 2px solid #ddd;
        text-align: center;
        font-size: 20px;
        font-weight: 600;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ===============================
# 🏷️ App Title
# ===============================
st.markdown("<div class='title'>📦 Amazon Review Sentiment Analyzer</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Analisis sentimen dari ulasan produk menggunakan Machine Learning</div>", unsafe_allow_html=True)

# ===============================
# 📝 Input User
# ===============================
text_input = st.text_area("Masukkan review produk:", height=150)

# Fitur Tambahan: contoh otomatis
if st.button("🔄 Gunakan contoh review"):
    text_input = "The product is broken and stopped working in 2 days"

# ===============================
# 🔍 Predict Button
# ===============================
if st.button("🔮 Prediksi Sentimen"):
    if text_input.strip() == "":
        st.warning("Masukkan teks review terlebih dahulu.")
    else:
        cleaned = clean_text(text_input)
        vector = tfidf.transform([cleaned])
        pred = model.predict(vector)[0]

        # warna output
        color_map = {
            "positive": "#4CAF50",
            "neutral": "#FFC107",
            "negative": "#F44336"
        }

        st.markdown(
            f"<div class='result-box' style='color:{color_map[pred]};'>Hasil Sentimen: {pred.upper()}</div>",
            unsafe_allow_html=True
        )

# ===============================
# 📊 Additional Feature: 
#    Menampilkan Confidence Model
# ===============================
if text_input.strip() != "":
    cleaned = clean_text(text_input)
    vector = tfidf.transform([cleaned])

    try:
        proba = model.predict_proba(vector)[0]
        st.subheader("📈 Confidence Level")
        st.write({
            "Negative": round(proba[0], 3),
            "Neutral": round(proba[1], 3),
            "Positive": round(proba[2], 3)
        })
    except:
        st.info("Model ini tidak mendukung probability (SVM). Confidence hanya muncul untuk Logistic Regression.")

# ===============================
# ℹ️ About Section
# ===============================
with st.expander("ℹ Tentang Aplikasi"):
    st.write("""
        Aplikasi ini menggunakan **Sentiment Analysis** berdasarkan dataset Amazon.
        
        Model terbaik dipilih dari **3 model machine learning**:
        - Logistic Regression  
        - SVM  
        - Random Forest  

        Aplikasi dibuat untuk Final Project CAMP Batch 3 — Data Science & GenAI.
    """)
