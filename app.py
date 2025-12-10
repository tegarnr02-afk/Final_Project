# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import random
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.utils.extmath import softmax as sk_softmax

st.set_page_config(page_title="Amazon Review Sentiment", layout="wide")

# ----------------------
# Session state init
# ----------------------
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"   # default theme
if "review_box" not in st.session_state:
    st.session_state["review_box"] = ""

# ----------------------
# Theme toggle helpers
# ----------------------
def switch_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

# Simple gradient toggle buttons (visual) - use on_click to avoid query_params hack
col_t1, col_t2, _ = st.columns([1,1,8])
with col_t1:
    st.button("🌙 Night", key="btn_night", on_click=lambda: st.session_state.update({"theme":"dark"}))
with col_t2:
    st.button("🌞 Day", key="btn_day", on_click=lambda: st.session_state.update({"theme":"light"}))

# Theme CSS
DARK_THEME = """
<style>
.stApp { background-color: #0f1720 !important; color: #e6eef8 !important; }
textarea, input, .stTextArea textarea, .stTextInput input { background:#11131a !important; color:#e6eef8 !important; }
.stButton>button { background:#11131a !important; color:#e6eef8 !important; border:1px solid #333 !important; }
.big-title { color: #fff !important; }
.sub { color: #b9c4d9 !important; }
</style>
"""
LIGHT_THEME = """
<style>
.stApp { background-color: #ffffff !important; color: #111 !important; }
textarea, input, .stTextArea textarea, .stTextInput input { background:#f5f5f5 !important; color:#111 !important; }
.stButton>button { background:#f0f0f0 !important; color:#111 !important; border:1px solid #ddd !important; }
.big-title { color: #111 !important; }
.sub { color: #555 !important; }
</style>
"""

st.markdown(DARK_THEME if st.session_state.theme == "dark" else LIGHT_THEME, unsafe_allow_html=True)

# ----------------------
# General style + header
# ----------------------
st.markdown(
    """
    <style>
    .big-title { font-size:34px; font-weight:700; margin-bottom:4px; }
    .sub { margin-bottom:18px; }
    .card { padding:12px; border-radius:10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns([2,1])
with col1:
    st.markdown('<div class="big-title">Amazon Review Sentiment Analysis</div>', unsafe_allow_html=True)
    st.markdown("<div class='sub'>Masukkan review atau upload CSV berisi kolom review untuk mendapatkan prediksi sentimen.</div>", unsafe_allow_html=True)
with col2:
    st.image("https://www.freeiconspng.com/uploads/amazon-icon-6.png", width=90)

st.markdown("---")

# ----------------------
# Stopwords & cleaning
# ----------------------
try:
    import nltk
    from nltk.corpus import stopwords
    nltk.data.find("corpora/stopwords")
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    STOPWORDS = set(ENGLISH_STOP_WORDS)

def clean_text(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = t.lower()
    t = re.sub(r"[^a-zA-Z\s]", " ", t)
    tokens = [w for w in t.split() if (w not in STOPWORDS) and len(w) > 1]
    return " ".join(tokens)

# ----------------------
# Prob + predict wrapper
# ----------------------
def get_proba_and_pred(model, X_vector):
    # determine classes if available
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
    else:
        classes = ["negative", "neutral", "positive"]
    # try predict_proba
    try:
        probs = model.predict_proba(X_vector)
    except Exception:
        try:
            scores = model.decision_function(X_vector)
            if scores.ndim == 1:
                scores = np.vstack([-scores, scores]).T
            probs = sk_softmax(scores, copy=True)
        except Exception:
            preds = model.predict(X_vector)
            probs = np.zeros((len(preds), len(classes)))
            for i, p in enumerate(preds):
                if p in classes:
                    probs[i, classes.index(p)] = 1.0
            probs[:] = 1.0 / len(classes)  # fallback uniform
    # adapt prob shape if mismatch
    if probs.shape[1] != len(classes):
        new = np.zeros((probs.shape[0], len(classes)))
        m = min(new.shape[1], probs.shape[1])
        new[:, :m] = probs[:, :m]
        new = new + 1e-6
        new = new / new.sum(axis=1, keepdims=True)
        probs = new
    preds = [classes[i] for i in probs.argmax(axis=1)]
    return probs, preds, classes

# ----------------------
# Load model & tfidf (local or upload)
# ----------------------
@st.cache_resource
def load_pickle_bytes(fobj):
    return pickle.loads(fobj.read())

model = None
tfidf = None

with st.sidebar.expander("Model / Vectorizer"):
    st.write("Upload model.pkl & tfidf.pkl (pickle) atau letakkan di folder app.")
    uploaded_model = st.file_uploader("Upload model.pkl", type=["pkl","pickle"], key="mdl_up")
    uploaded_tfidf = st.file_uploader("Upload tfidf.pkl", type=["pkl","pickle"], key="tfidf_up")
    use_local = st.checkbox("Muat model.pkl & tfidf.pkl dari folder aplikasi (jika ada)", value=True)

if use_local:
    try:
        with open("model.pkl","rb") as f:
            model = pickle.load(f)
    except Exception:
        model = None
    try:
        with open("tfidf.pkl","rb") as f:
            tfidf = pickle.load(f)
    except Exception:
        tfidf = None

if uploaded_model is not None:
    try:
        model = load_pickle_bytes(uploaded_model)
        st.sidebar.success("Model diunggah")
    except Exception as e:
        st.sidebar.error(f"Gagal load model: {e}")

if uploaded_tfidf is not None:
    try:
        tfidf = load_pickle_bytes(uploaded_tfidf)
        st.sidebar.success("TF-IDF diunggah")
    except Exception as e:
        st.sidebar.error(f"Gagal load tfidf: {e}")

if model is None or tfidf is None:
    st.sidebar.info("Jika belum ada model, jalankan training di Colab lalu unduh 'model.pkl' & 'tfidf.pkl'.")

# ----------------------
# Main UI - single input
# ----------------------
st.markdown("### Masukkan review produk:")
colA, colB = st.columns([4,1])

with colA:
    text_input = st.text_area("Masukkan review di sini...", value=st.session_state["review_box"], height=140, key="review_input",
                              placeholder="Contoh: The product stopped working after 2 days. Very disappointed.")

with colB:
    st.write("Aksi:")
    if st.button("Reset", key="btn_reset"):
        st.session_state["review_box"] = ""
        st.experimental_rerun()

    # contoh acak
    sample_examples = {
        "positive": [
            "Amazing product! Exceeded my expectations.",
            "Works perfectly, very satisfied!",
            "High quality and fast delivery."
        ],
        "neutral": [
            "The product is okay, nothing special.",
            "Average experience, not bad not great.",
            "It's fine for the price."
        ],
        "negative": [
            "Terrible quality. Very disappointed.",
            "Stopped working after two days.",
            "Not worth the money at all."
        ]
    }

    if st.button("Contoh Acak", key="btn_random"):
        all_reviews = sample_examples["positive"] + sample_examples["neutral"] + sample_examples["negative"]
        choice = random.choice(all_reviews)
        st.session_state["review_box"] = choice
        st.experimental_rerun()

# ensure textbox shows session value after buttons
if st.session_state["review_box"]:
    # update textarea to session value (no duplicate element id because we used keys)
    text_input = st.session_state["review_box"]

# Predict single
if st.button("Prediksi Sentimen", key="btn_predict"):
    if model is None or tfidf is None:
        st.error("Model atau TF-IDF belum tersedia. Upload atau letakkan file model.pkl & tfidf.pkl.")
    elif not isinstance(text_input, str) or text_input.strip() == "":
        st.error("Masukkan teks review terlebih dahulu.")
    else:
        cleaned = clean_text(text_input)
        try:
            vec = tfidf.transform([cleaned])
        except Exception as e:
            st.error(f"Error saat transform: {e}")
            st.stop()
        try:
            probs, preds, classes = get_proba_and_pred(model, vec)
            pred = preds[0]
            proba_map = {classes[i]: float(probs[0, i]) for i in range(len(classes))}
            st.markdown(f"**Prediksi Sentimen:** `{pred}`")
            st.markdown("**Confidence:**")
            dfc = pd.DataFrame.from_dict(proba_map, orient="index", columns=["probability"]).sort_values("probability", ascending=False)
            st.table((dfc*100).round(2))
            st.bar_chart(dfc["probability"])
        except Exception as e:
            st.error(f"Gagal memprediksi: {e}")

st.markdown("---")

# ----------------------
# Batch upload & prediksi
# ----------------------
st.markdown("## Prediksi Batch (Upload CSV)")
uploaded_csv = st.file_uploader("Upload CSV untuk batch prediksi", type=["csv"], key="batch_up")
if uploaded_csv is not None:
    try:
        df_upload = pd.read_csv(uploaded_csv)
        st.write("Preview data:")
        st.dataframe(df_upload.head())
        col_options = list(df_upload.columns)
        chosen_col = st.selectbox("Pilih kolom yang berisi review", col_options, key="col_choice")
        n_preview = st.number_input("Jumlah baris preview dan prediksi", min_value=1, max_value=500, value=10, key="num_preview")
        if st.button("Jalankan prediksi batch", key="btn_batch"):
            if model is None or tfidf is None:
                st.error("Model / TF-IDF belum tersedia.")
            else:
                texts = df_upload[chosen_col].fillna("").astype(str).tolist()
                cleaned_texts = [clean_text(t) for t in texts]
                try:
                    X_vec = tfidf.transform(cleaned_texts)
                except Exception as e:
                    st.error(f"Error saat transform batch: {e}")
                    st.stop()
                probs, preds, classes = get_proba_and_pred(model, X_vec)
                proba_df = pd.DataFrame(probs, columns=[f"prob_{c}" for c in classes])
                out = df_upload.copy().reset_index(drop=True)
                out["pred_sentiment"] = preds
                out = pd.concat([out, proba_df], axis=1)
                st.success("Selesai. Preview hasil:")
                st.dataframe(out.head(n_preview))
                csv_bytes = out.to_csv(index=False).encode("utf-8")
                st.download_button("Download hasil prediksi (CSV)", csv_bytes, "prediksi_hasil.csv", "text/csv")
    except Exception as e:
        st.error(f"Gagal membaca file CSV: {e}")

st.markdown("---")

# ----------------------
# Optional EDA (wordcloud, distribution)
# ----------------------
st.markdown("## Exploratory (Optional)")
with st.expander("Upload sample CSV untuk EDA (sama file yg dipakai di training)"):
    sample_file = st.file_uploader("Upload sample dataset untuk EDA", type=["csv"], key="eda_up")
    if sample_file is not None:
        try:
            df_s = pd.read_csv(sample_file)
            st.write("Preview:")
            st.dataframe(df_s.head())
            possible = [c for c in df_s.columns if any(k in c.lower() for k in ["review","text","comment"])]
            chosen = st.selectbox("Pilih kolom teks untuk EDA", possible if possible else df_s.columns, key="eda_col")
            if model is not None and tfidf is not None:
                texts = df_s[chosen].fillna("").astype(str).tolist()
                cleaned_texts = [clean_text(t) for t in texts]
                X_vec = tfidf.transform(cleaned_texts)
                probs, preds, classes = get_proba_and_pred(model, X_vec)
                df_s["pred_sentiment"] = preds
                st.write("Distribusi prediksi:")
                st.bar_chart(df_s["pred_sentiment"].value_counts())
            text_all = " ".join(df_s[chosen].fillna("").astype(str).tolist())
            wc = WordCloud(width=800, height=400, background_color="white").generate(text_all)
            fig, ax = plt.subplots(figsize=(10,4))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error EDA: {e}")

st.markdown("---")
st.write("About: Demo final project. Pastikan model.pkl & tfidf.pkl cocok (TF-IDF harus sama saat training & inference).")
