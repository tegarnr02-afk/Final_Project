import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.extmath import softmax as sk_softmax

st.set_page_config(page_title="Amazon Review Sentiment", layout="wide")

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def switch_theme():
    if st.session_state.theme == "dark":
        st.session_state.theme = "light"
    else:
        st.session_state.theme = "dark"



def toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

# Custom CSS tombol toggle
toggle_css = """
<style>
.toggle-wrapper {
    display: flex;
    gap: 20px;
    margin-bottom: 20px;
}

.toggle-btn {
    width: 160px;
    height: 55px;
    border-radius: 40px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    cursor: pointer;
    padding: 0px 15px;
    font-weight: 600;
    font-size: 16px;
}

.day {
    background: linear-gradient(45deg, #ff6b81, #feca57);
    color: white;
    border: 3px solid #ffa502;
}

.night {
    background: linear-gradient(45deg, #1e3799, #4a69bd);
    color: white;
    border: 3px solid #0c2461;
}

.icon-circle {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
}
</style>
"""

st.markdown(toggle_css, unsafe_allow_html=True)

# Buat tombol toggle
day_clicked = st.button(" ", key="daybtn")
night_clicked = st.button(" ", key="nightbtn")

toggle_html = f"""
<div class="toggle-wrapper">

<div class="toggle-btn day" onclick="window.location.href='?theme=light'">
    DAY MODE
    <div class="icon-circle">☀</div>
</div>

<div class="toggle-btn night" onclick="window.location.href='?theme=dark'">
    NIGHT MODE
    <div class="icon-circle">🌙</div>
</div>

</div>
"""

st.markdown(toggle_html, unsafe_allow_html=True)

if "theme" in st.query_params:
    st.session_state.theme = st.query_params["theme"]
    st.query_params.clear()
    st.rerun()



LIGHT_THEME = """
<style>
.stApp { background-color: #ffffff !important; color: #000000 !important; }
textarea, input, .stTextArea textarea {
    background:#f3f3f3 !important; color:#000000 !important;
}
</style>
"""

DARK_THEME = """
<style>
.stApp { background-color: #0f1720 !important; color: #e6eef8 !important; }
textarea, input, .stTextArea textarea {
    background:#11131a !important; color:#e6eef8 !important;
}
</style>
"""

if st.session_state.theme == "dark":
    st.markdown(DARK_THEME, unsafe_allow_html=True)
else:
    st.markdown(LIGHT_THEME, unsafe_allow_html=True)





# CSS untuk tema

# --- Style ---
st.markdown(
    """
    <style>
   
    .big-title { font-size:36px; font-weight:700; color: #ffffff; }
    .sub { color: #b9c4d9; }
    .card { padding:12px; border-radius:10px; background:#11131a; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Helper: stopwords fallback ---
try:
    import nltk
    from nltk.corpus import stopwords
    nltk.data.find("corpora/stopwords")
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    # fallback if nltk data not available (use sklearn's stopwords)
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    STOPWORDS = set([w for w in ENGLISH_STOP_WORDS])

# --- Text cleaning function (consistent with Colab) ---
def clean_text(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = t.lower()
    t = re.sub(r"[^a-zA-Z\s]", " ", t)
    tokens = [w for w in t.split() if (w not in STOPWORDS) and (len(w) > 1)]
    return " ".join(tokens)

# --- Utility: predict_proba wrapper (works even if model has no predict_proba) ---
def get_proba_and_pred(model, X_vector):
    classes = None
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
    else:
        # fallback generic classes (must correspond to training)
        classes = ["negative", "neutral", "positive"]

    # try predict_proba
    try:
        probs = model.predict_proba(X_vector)
    except Exception:
        # try decision_function -> softmax
        try:
            scores = model.decision_function(X_vector)
            # ensure 2D
            if scores.ndim == 1:
                # binary: convert to two-class probabilities
                scores = np.vstack([-scores, scores]).T
            probs = sk_softmax(scores, copy=True)
        except Exception:
            # as last resort, use one-hot from predict()
            preds = model.predict(X_vector)
            probs = np.zeros((len(preds), len(classes)))
            for i, p in enumerate(preds):
                if p in classes:
                    probs[i, classes.index(p)] = 1.0
            # if class labels mismatch length, pad/truncate
            if probs.shape[1] != len(classes):
                probs = np.zeros((len(preds), len(classes)))
                # fill equal low prob
                probs[:] = 1.0 / len(classes)
    # ensure prob array shape matches classes length
    if probs.shape[1] != len(classes):
        # try to adapt shape by padding/trunc
        new = np.zeros((probs.shape[0], len(classes)))
        m = min(new.shape[1], probs.shape[1])
        new[:, :m] = probs[:, :m]
        if new.sum(axis=1).min() == 0:
            new = new + 1e-6
        new = new / new.sum(axis=1, keepdims=True)
        probs = new
    preds = [classes[i] for i in probs.argmax(axis=1)]
    return probs, preds, classes

# --- Load model & tfidf (try local files first) ---
@st.cache_resource
def load_pickle(path_bytes):
    return pickle.loads(path_bytes.read())

model = None
tfidf = None

col1, col2 = st.columns([2,1])
with col1:
    st.markdown('<div class="big-title">Amazon Review Sentiment Analysis</div>', unsafe_allow_html=True)
    st.markdown("<div class='sub'>Masukkan review atau upload CSV berisi kolom review untuk mendapatkan prediksi sentimen.</div>", unsafe_allow_html=True)

with col2:
    st.image("https://www.freeiconspng.com/uploads/amazon-icon-6.png", width=90)

st.markdown("---")

# Sidebar: load/upload models
with st.sidebar.expander("Model / Vectorizer"):
    st.write("Model & TF-IDF harus tersedia. Pilih salah satu:")
    uploaded_model = st.file_uploader("Upload model.pkl (pickle)", type=["pkl","pickle"], key="m1")
    uploaded_tfidf = st.file_uploader("Upload tfidf.pkl (pickle)", type=["pkl","pickle"], key="v1")
    use_local = st.checkbox("Gunakan model.pkl & tfidf.pkl dari folder aplikasi (jika ada)", value=True)

# try loading local files if asked
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

# override with uploads if provided
if uploaded_model is not None:
    try:
        model = load_pickle(uploaded_model)
        st.sidebar.success("Model ter-upload")
    except Exception as e:
        st.sidebar.error(f"Gagal load model: {e}")

if uploaded_tfidf is not None:
    try:
        tfidf = load_pickle(uploaded_tfidf)
        st.sidebar.success("TF-IDF ter-upload")
    except Exception as e:
        st.sidebar.error(f"Gagal load tfidf: {e}")

if model is None or tfidf is None:
    st.warning("Model atau TF-IDF belum tersedia. Unggah keduanya atau letakkan model.pkl & tfidf.pkl di folder aplikasi.")
    st.info("Jika belum punya, jalankan training di Colab lalu unduh model.pkl dan tfidf.pkl menggunakan pickle.dump.")

# --- Main UI: single review input + reset button ---
st.markdown("### Masukkan review produk:")

if "review_box" not in st.session_state:
    st.session_state["review_box"] = ""

colA, colB = st.columns([4,1])

with colA:
    text_input = st.text_area(
        "Masukkan review di sini...",
        height=140,
        key="review_box",
        placeholder="Contoh: The product stopped working after 2 days. Very disappointed."
    )

with colB:
    st.write("Aksi:")

    # tombol reset
    def reset_input():
        st.session_state["review_box"] = ""

    st.button("Reset", on_click=reset_input)

    # tombol contoh random
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

    import random

    def random_example():
        all_reviews = (
            sample_examples["positive"]
            + sample_examples["neutral"]
            + sample_examples["negative"]
        )
        st.session_state["review_box"] = random.choice(all_reviews)

    st.button("Contoh Acak", on_click=random_example)

# Button to predict single review
if st.button("Prediksi Sentimen"):
    if model is None or tfidf is None:
        st.error("Model atau TF-IDF belum tersedia. Upload atau tempatkan file model.pkl dan tfidf.pkl.")
    else:
        if not isinstance(text_input, str) or text_input.strip() == "":
            st.error("Masukkan teks review terlebih dahulu.")
        else:
            cleaned = clean_text(text_input)
            try:
                vec = tfidf.transform([cleaned])
            except Exception as e:
                st.error(f"Terjadi error saat melakukan transform: {e}")
                st.stop()

            try:
                probs, preds, classes = get_proba_and_pred(model, vec)
                pred = preds[0]
                proba_map = {classes[i]: float(probs[0,i]) for i in range(len(classes))}
                st.markdown(f"*Prediksi Sentimen:* {pred}")
                st.markdown("*Confidence:*")
                dfc = pd.DataFrame.from_dict(proba_map, orient="index", columns=["probability"]).sort_values("probability", ascending=False)
                st.table((dfc*100).round(2))
                # simple bar chart
                st.bar_chart(dfc["probability"])
            except Exception as e:
                st.error(f"Gagal memprediksi: {e}")

st.markdown("---")

# --- Batch prediction via CSV upload ---
st.markdown("## Prediksi Batch (Upload CSV)")
st.write("Upload CSV berisi kolom teks review (contoh header: 'review_content', 'text', 'review').")

uploaded_csv = st.file_uploader("Upload CSV untuk batch prediksi", type=["csv"])
if uploaded_csv is not None:
    try:
        df_upload = pd.read_csv(uploaded_csv)
        st.write("Preview data:")
        st.dataframe(df_upload.head())
        # choose column
        col_options = list(df_upload.columns)
        chosen_col = st.selectbox("Pilih kolom yang berisi review", col_options)
        n_preview = st.number_input("Jumlah baris preview dan prediksi", min_value=1, max_value=500, value=10)
        if st.button("Jalankan prediksi batch"):
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
                # download
                csv_bytes = out.to_csv(index=False).encode("utf-8")
                st.download_button("Download hasil prediksi (CSV)", csv_bytes, "prediksi_hasil.csv", "text/csv")
    except Exception as e:
        st.error(f"Gagal membaca file CSV: {e}")

st.markdown("---")

# --- EDA (if sample dataset available) ---
st.markdown("## Exploratory (Optional)")
with st.expander("Upload sample CSV untuk EDA (sama file yg dipakai di training)"):
    sample_file = st.file_uploader("Upload sample dataset untuk EDA", type=["csv"], key="eda")
    if sample_file is not None:
        try:
            df_s = pd.read_csv(sample_file)
            st.write("Preview:")
            st.dataframe(df_s.head())
            # try pick text column automatically
            possible = [c for c in df_s.columns if any(k in c.lower() for k in ["review","text","comment"])]
            if len(possible) == 0:
                chosen = st.selectbox("Pilih kolom teks untuk EDA", df_s.columns)
            else:
                chosen = st.selectbox("Pilih kolom teks untuk EDA", possible)
            st.write("Menghitung distribution sentiment (menggunakan model jika tersedia)...")
            if model is not None and tfidf is not None:
                texts = df_s[chosen].fillna("").astype(str).tolist()
                cleaned_texts = [clean_text(t) for t in texts]
                X_vec = tfidf.transform(cleaned_texts)
                probs, preds, classes = get_proba_and_pred(model, X_vec)
                df_s["pred_sentiment"] = preds
                st.write("Distribusi prediksi:")
                st.bar_chart(df_s["pred_sentiment"].value_counts())
            # wordcloud of column
            text_all = " ".join(df_s[chosen].fillna("").astype(str).tolist())
            wc = WordCloud(width=800, height=400, background_color="white").generate(text_all)
            fig, ax = plt.subplots(figsize=(10,4))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error EDA: {e}")

st.markdown("---")
st.write("About: Aplikasi ini untuk demo final project. Pastikan model.pkl & tfidf.pkl cocok (dilatih dengan TF-IDF yang sama).")
