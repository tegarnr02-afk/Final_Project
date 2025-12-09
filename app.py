import streamlit as st
import pickle
import re

# ================================
# LOAD MODEL & VECTOR
# ================================
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# ================================
# CLEANING TEXT (Tanpa NLTK, aman)
# ================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', "", text)
    return text

# ================================
# CUSTOM STYLE CSS
# ================================
st.markdown("""
    <style>
    body {
        background-color: #F7F7F7;
    }
    .title {
        font-size: 32px;
        font-weight: bold;
        color: #4F46E5;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtext {
        text-align: center;
        color: #555;
        margin-bottom: 25px;
    }
    .card {
        background-color: white;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        border-left: 6px solid #4F46E5;
    }
    .example-box {
        background-color: #EEF2FF;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ================================
# TITLE
# ================================
st.markdown("<h1 class='title'>Amazon Product Review Sentiment App</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtext'>Masukkan review produk untuk mengetahui sentimennya.</p>", unsafe_allow_html=True)

# ================================
# EXAMPLE REVIEWS
# ================================
with st.expander("📌 Contoh Review Otomatis (Klik untuk memilih)"):
    
    col1, col2, col3 = st.columns(3)

    if col1.button("😊 Review Positive"):
        st.session_state["example"] = "Amazing product! Works perfectly and worth the money."

    if col2.button("😐 Review Neutral"):
        st.session_state["example"] = "The product is okay, nothing special but works fine."

    if col3.button("😡 Review Negative"):
        st.session_state["example"] = "Terrible quality. Broke after one use. Not recommended."

# Input area
default_text = st.session_state.get("example", "")
text_input = st.text_area("📝 Tulis Review Anda:", value=default_text, height=160)

# ================================
# BUTTON PREDICT
# ================================
if st.button("🔍 Prediksi Sentimen"):

    if text_input.strip() == "":
        st.warning("⚠️ Silakan masukkan teks review terlebih dahulu.")
    else:
        cleaned = clean_text(text_input)
        vector = tfidf.transform([cleaned])
        prediction = model.predict(vector)[0]

        # ================================
        # OUTPUT CARD
        # ================================
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        if prediction == "positive":
            st.success("😊 **Sentimen: POSITIVE**")
            st.write("Review menunjukkan bahwa pelanggan merasa puas.")
        elif prediction == "neutral":
            st.info("😐 **Sentimen: NEUTRAL**")
            st.write("Review bernada biasa saja, tidak terlalu puas atau kecewa.")
        else:
            st.error("😡 **Sentimen: NEGATIVE**")
            st.write("Review bernada kecewa terhadap produk.")

        st.markdown("</div>", unsafe_allow_html=True)
