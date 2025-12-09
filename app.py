import streamlit as st
import pickle
import re
import pandas as pd

# =========================
# LOAD MODEL & TFIDF
# =========================
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# =========================
# TEXT CLEANING FUNCTION
# =========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text


# =========================
# SINGLE TEXT PREDICTION
# =========================
def predict_sentiment(text):
    cleaned = clean_text(text)
    vector = tfidf.transform([cleaned])
    pred = model.predict(vector)[0]

    # probability / confidence
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vector)[0]
        return pred, {
            "Negative": round(float(probs[0]), 3),
            "Neutral": round(float(probs[1]), 3),
            "Positive": round(float(probs[2]), 3),
        }
    else:
        # fallback jika model tidak punya predict_proba
        return pred, {"Negative": "-", "Neutral": "-", "Positive": "-"}


# =========================
# STREAMLIT PAGE SETTING
# =========================
st.set_page_config(
    page_title="Amazon Sentiment Analysis",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Amazon Review Sentiment Analysis")
st.write("Analisis sentimen dari ulasan produk menggunakan Machine Learning.")

# =========================
# EXAMPLE REVIEWS
# =========================
example_reviews = {
    "Positive": "This product is amazing! Great quality and totally worth the price.",
    "Neutral": "The product is okay, nothing special. It works but not too impressive.",
    "Negative": "Very disappointed. The product broke after one use, terrible quality."
}

st.subheader("Masukkan review produk:")

col1, col2 = st.columns([3,1])
with col1:
    text_input = st.text_area("", height=160)

with col2:
    if st.button("🔄 Gunakan contoh review"):
        st.session_state['example'] = example_reviews
        st.write("### Contoh Review:")
        st.write("🟢 **Positive:**", example_reviews["Positive"])
        st.write("⚪ **Neutral:**", example_reviews["Neutral"])
        st.write("🔴 **Negative:**", example_reviews["Negative"])

# =========================
# PREDICTION BUTTON
# =========================
if st.button("🔍 Prediksi Sentimen"):
    if text_input.strip() == "":
        st.warning("Masukkan review terlebih dahulu.")
    else:
        sentiment, confidence = predict_sentiment(text_input)
        st.success(f"**Hasil Sentimen: {sentiment}**")

        st.subheader("📊 Confidence Level")
        st.json(confidence)


# =========================
# CSV UPLOAD SECTION
# =========================
st.markdown("---")
st.subheader("📁 Upload CSV untuk Prediksi Banyak Review")

uploaded_file = st.file_uploader("Upload file CSV (kolom wajib: text)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "text" not in df.columns:
        st.error("CSV harus memiliki kolom bernama **text**.")
    else:
        st.write("### 5 Data Teratas")
        st.dataframe(df.head())

        # Prediksi
        df["clean_text"] = df["text"].apply(clean_text)
        vectors = tfidf.transform(df["clean_text"])
        df["predicted_sentiment"] = model.predict(vectors)

        st.write("### Hasil Prediksi")
        st.dataframe(df[["text", "predicted_sentiment"]])

        # Download hasil
        csv_output = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇ Download Hasil Prediksi CSV",
            data=csv_output,
            file_name="hasil_prediksi.csv",
            mime="text/csv"
        )

# =========================
# FOOTER / ABOUT
# =========================
st.markdown("---")
with st.expander("ℹ Tentang Aplikasi"):
    st.write("""
        Aplikasi ini menggunakan:
        - **TF-IDF Vectorizer** untuk mengubah teks menjadi angka  
        - **Logistic Regression / SVM / RandomForest** (model terbaik dari eksperimen)
        - Dapat memprediksi satu review atau banyak review melalui file CSV  
        
        Dibuat untuk Final Project Data Science & Generative AI.
    """)

