import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import io
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.extmath import softmax as sk_softmax

st.set_page_config(page_title="Amazon Review Sentiment", layout="wide")

# ============================
# THEME TOGGLE SYSTEM
# ============================
if "theme" not in st.session_state:
    st.session_state.theme = "dark"



# ===========================
# THEME STYLES
# ===========================
if st.session_state.theme == "light":
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #ffffff !important;
    }
    [data-testid="stApp"] {
        background-color: #ffffff !important;
    }
    [data-testid="stSidebar"] {
        background-color: #f5f5f5 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #000000 !important;
    }
    .block-container {
        background-color: #ffffff !important;
    }
    .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6, .stMarkdown span, .stMarkdown div,
    label, .stSelectbox label, .stTextInput label, .stTextArea label {
        color: #000000 !important;
    }
    [data-testid="stDataFrame"], .dataframe {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    .dataframe th {
        background-color: #f0f0f0 !important;
        color: #000000 !important;
    }
    .dataframe td {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    .stButton > button {
        background-color: #ffd700 !important;
        color: #000000 !important;
        border: 1px solid #cccccc !important;
    }
    .stButton > button:hover {
        background-color: #ffed4e !important;
        border: 1px solid #999999 !important;
    }
    textarea, input {
        background-color: #f9f9f9 !important;
        color: #000000 !important;
        border: 1px solid #cccccc !important;
    }
    [data-testid="stExpander"] {
        background-color: #f9f9f9 !important;
        border: 1px solid #e0e0e0 !important;
    }
    [data-testid="stExpander"] summary {
        color: #000000 !important;
    }
    [data-testid="stFileUploader"] {
        background-color: #f9f9f9 !important;
    }
    [data-testid="stFileUploader"] label {
        color: #000000 !important;
    }
    [data-testid="stMetricValue"] {
        color: #000000 !important;
    }
    </style>
    """, unsafe_allow_html=True)
else:  # dark theme
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #0e1117 !important;
    }
    [data-testid="stApp"] {
        background-color: #0e1117 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #1a1d24 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #fafafa !important;
    }
    .block-container {
        background-color: #0e1117 !important;
    }
    .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6, .stMarkdown span, .stMarkdown div,
    label, .stSelectbox label, .stTextInput label, .stTextArea label {
        color: #fafafa !important;
    }
    [data-testid="stDataFrame"], .dataframe {
        background-color: #1a1d24 !important;
        color: #fafafa !important;
    }
    .dataframe th {
        background-color: #262a33 !important;
        color: #fafafa !important;
    }
    .dataframe td {
        background-color: #1a1d24 !important;
        color: #fafafa !important;
    }
    .stButton > button {
        background-color: #262a33 !important;
        color: #fafafa !important;
        border: 1px solid #3d4450 !important;
    }
    .stButton > button:hover {
        background-color: #2d323d !important;
        border: 1px solid #4d5562 !important;
    }
    textarea, input {
        background-color: #1a1d24 !important;
        color: #fafafa !important;
        border: 1px solid #3d4450 !important;
    }
    [data-testid="stExpander"] {
        background-color: #1a1d24 !important;
        border: 1px solid #3d4450 !important;
    }
    [data-testid="stExpander"] summary {
        color: #fafafa !important;
    }
    [data-testid="stFileUploader"] {
        background-color: #1a1d24 !important;
    }
    [data-testid="stFileUploader"] label {
        color: #fafafa !important;
    }
    [data-testid="stMetricValue"] {
        color: #fafafa !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ===========================
# HELPER FUNCTIONS
# ===========================
try:
    import nltk
    from nltk.corpus import stopwords
    nltk.data.find("corpora/stopwords")
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    STOPWORDS = set([w for w in ENGLISH_STOP_WORDS])

def clean_text(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = t.lower()
    t = re.sub(r"[^a-zA-Z\s]", " ", t)
    tokens = [w for w in t.split() if (w not in STOPWORDS) and (len(w) > 1)]
    return " ".join(tokens)

def get_proba_and_pred(model, X_vector):
    classes = None
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
    else:
        classes = ["negative", "neutral", "positive"]

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
            if probs.shape[1] != len(classes):
                probs = np.zeros((len(preds), len(classes)))
                probs[:] = 1.0 / len(classes)
    
    if probs.shape[1] != len(classes):
        new = np.zeros((probs.shape[0], len(classes)))
        m = min(new.shape[1], probs.shape[1])
        new[:, :m] = probs[:, :m]
        if new.sum(axis=1).min() == 0:
            new = new + 1e-6
        new = new / new.sum(axis=1, keepdims=True)
        probs = new
    preds = [classes[i] for i in probs.argmax(axis=1)]
    return probs, preds, classes

@st.cache_resource
def load_pickle(path_bytes):
    return pickle.loads(path_bytes.read())

# ===========================
# LOAD MODELS
# ===========================
model = None
tfidf = None

col1, col2 = st.columns([3,1])
with col1:
    st.markdown('<h1>🛒 Amazon Review Sentiment Analysis</h1>', unsafe_allow_html=True)
    st.markdown("Masukkan review atau upload CSV berisi kolom review untuk mendapatkan prediksi sentimen.")

with col2:
    # Logo dan toggle dalam satu kolom
    subcol1, subcol2 = st.columns([1, 1])
    with subcol1:
        st.image("https://www.freeiconspng.com/uploads/amazon-icon-6.png", width=70)
    with subcol2:
        # Compact toggle HTML
        toggle_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
        <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }}
        
        .theme-toggle-container {{
            display: flex;
            justify-content: flex-start;
            padding: 0;
            background: transparent;
        }}
        
        .toggle-wrapper {{
            position: relative;
            width: 140px;
            height: 36px;
            background: linear-gradient(90deg, #ff6b9d 0%, #ffa06b 50%, #ffd93d 100%);
            border-radius: 25px;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(255, 107, 157, 0.3);
            transition: all 0.3s cubic-bezier(0.68, -0.55, 0.265, 1.55);
            overflow: hidden;
            user-select: none;
        }}
        
        .toggle-wrapper.night-mode {{
            background: linear-gradient(90deg, #2a5298 0%, #1e3c72 50%, #4facfe 100%);
            box-shadow: 0 4px 12px rgba(79, 172, 254, 0.3);
        }}
        
        .toggle-slider {{
            position: absolute;
            top: 3px;
            left: 3px;
            width: 62px;
            height: 30px;
            background: white;
            border-radius: 25px;
            transition: all 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55);
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            font-size: 9px;
            color: #ff6b9d;
            letter-spacing: 0.3px;
            z-index: 2;
        }}
        
        .toggle-wrapper.night-mode .toggle-slider {{
            left: 75px;
            color: #2a5298;
        }}
        
        .toggle-option {{
            position: absolute;
            top: 50%;
            transform: translateY(-50%);
            display: flex;
            align-items: center;
            gap: 4px;
            font-weight: 600;
            font-size: 9px;
            color: white;
            transition: all 0.3s ease;
            z-index: 1;
            pointer-events: none;
        }}
        
        .toggle-option-left {{
            left: 10px;
        }}
        
        .toggle-option-right {{
            right: 10px;
        }}
        
        .toggle-wrapper.night-mode .toggle-option-left {{
            opacity: 0.6;
        }}
        
        .toggle-wrapper:not(.night-mode) .toggle-option-right {{
            opacity: 0.6;
        }}
        
        .toggle-icon {{
            font-size: 14px;
            line-height: 1;
        }}
        
        .sun-icon {{
            animation: rotate 20s linear infinite;
        }}
        
        .moon-icon {{
            animation: pulse 3s ease-in-out infinite;
        }}
        
        @keyframes rotate {{
            from {{ transform: rotate(0deg); }}
            to {{ transform: rotate(360deg); }}
        }}
        
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.7; }}
        }}
        
        .toggle-wrapper:hover {{
            transform: scale(1.05);
        }}
        
        .toggle-wrapper:active {{
            transform: scale(0.95);
        }}
        </style>
        </head>
        <body>
        <div class="theme-toggle-container">
            <div class="toggle-wrapper {'night-mode' if st.session_state.theme == 'dark' else ''}" id="toggleWrapper" onclick="toggleTheme()">
                <div class="toggle-slider" id="toggleSlider">
                    <span id="sliderText">{'NIGHT' if st.session_state.theme == 'dark' else 'DAY'}</span>
                </div>
                <div class="toggle-option toggle-option-left">
                    <span class="toggle-icon sun-icon">☀</span>
                    <span>DAY</span>
                </div>
                <div class="toggle-option toggle-option-right">
                    <span class="toggle-icon moon-icon">🌙</span>
                    <span>NIGHT</span>
                </div>
            </div>
        </div>
        
        <script>
        function toggleTheme() {{
            const wrapper = document.getElementById('toggleWrapper');
            const sliderText = document.getElementById('sliderText');
            
            // Toggle visual immediately
            if (wrapper.classList.contains('night-mode')) {{
                wrapper.classList.remove('night-mode');
                sliderText.textContent = 'DAY';
            }} else {{
                wrapper.classList.add('night-mode');
                sliderText.textContent = 'NIGHT';
            }}
            
            // Find and click Streamlit button
            setTimeout(() => {{
                try {{
                    const parentDoc = window.parent.document;
                    let themeButton = null;
                    
                    themeButton = parentDoc.querySelector('button[title="Toggle Theme"]');
                    
                    if (!themeButton) {{
                        const allButtons = parentDoc.querySelectorAll('button');
                        for (let btn of allButtons) {{
                            const btnText = btn.innerText || btn.textContent || '';
                            if (btnText.includes('🌓')) {{
                                themeButton = btn;
                                break;
                            }}
                        }}
                    }}
                    
                    if (themeButton) {{
                        themeButton.click();
                    }}
                }} catch(e) {{
                    console.error('Error toggling theme:', e);
                }}
            }}, 50);
        }}
        </script>
        </body>
        </html>
        """
        components.html(toggle_html, height=50, scrolling=False)

st.markdown("---")

# Sidebar: Model upload
with st.sidebar.expander("📦 Model / Vectorizer", expanded=True):
    st.write("Model & TF-IDF harus tersedia:")
    uploaded_model = st.file_uploader("Upload model.pkl", type=["pkl","pickle"], key="m1")
    uploaded_tfidf = st.file_uploader("Upload tfidf.pkl", type=["pkl","pickle"], key="v1")
    use_local = st.checkbox("Gunakan file lokal (model.pkl & tfidf.pkl)", value=True)

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
        model = load_pickle(uploaded_model)
        st.sidebar.success("✅ Model ter-upload")
    except Exception as e:
        st.sidebar.error(f"❌ Gagal load model: {e}")

if uploaded_tfidf is not None:
    try:
        tfidf = load_pickle(uploaded_tfidf)
        st.sidebar.success("✅ TF-IDF ter-upload")
    except Exception as e:
        st.sidebar.error(f"❌ Gagal load tfidf: {e}")

if model is None or tfidf is None:
    st.warning("⚠ Model atau TF-IDF belum tersedia. Unggah keduanya atau letakkan model.pkl & tfidf.pkl di folder aplikasi.")
    st.info("💡 Jika belum punya, jalankan training di Colab lalu unduh model.pkl dan tfidf.pkl.")


# ===========================
# SINGLE PREDICTION
# ===========================
st.markdown("### 📝 Masukkan review produk:")

if "review_box" not in st.session_state:
    st.session_state["review_box"] = ""

colA, colB = st.columns([4,1])

with colA:
    text_input = st.text_area(
        "Review:",
        height=140,
        key="review_box",
        placeholder="Contoh: The product stopped working after 2 days. Very disappointed."
    )

with colB:
    st.write("*Aksi:*")

    def reset_input():
        st.session_state["review_box"] = ""

    st.button("🔄 Reset", on_click=reset_input, use_container_width=True)

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

    st.button("🎲 Contoh", on_click=random_example, use_container_width=True)

if st.button("🔍 Prediksi Sentimen", use_container_width=True):
    if model is None or tfidf is None:
        st.error("❌ Model atau TF-IDF belum tersedia.")
    else:
        if not isinstance(text_input, str) or text_input.strip() == "":
            st.error("❌ Masukkan teks review terlebih dahulu.")
        else:
            cleaned = clean_text(text_input)
            try:
                vec = tfidf.transform([cleaned])
            except Exception as e:
                st.error(f"❌ Error saat transform: {e}")
                st.stop()

            try:
                probs, preds, classes = get_proba_and_pred(model, vec)
                pred = preds[0]
                proba_map = {classes[i]: float(probs[0,i]) for i in range(len(classes))}
                
                st.success(f"*Prediksi Sentimen:* {pred.upper()}")
                st.markdown("*Confidence:*")
                dfc = pd.DataFrame.from_dict(proba_map, orient="index", columns=["probability"]).sort_values("probability", ascending=False)
                st.table((dfc*100).round(2))
                st.bar_chart(dfc["probability"])
            except Exception as e:
                st.error(f"❌ Gagal memprediksi: {e}")

st.markdown("---")

# ===========================
# BATCH PREDICTION
# ===========================
st.markdown("## 📊 Prediksi Batch (Upload CSV)")
st.write("Upload CSV berisi kolom teks review.")

uploaded_csv = st.file_uploader("Upload CSV untuk batch prediksi", type=["csv"])
if uploaded_csv is not None:
    try:
        df_upload = pd.read_csv(uploaded_csv)
        st.write("*Preview data:*")
        st.dataframe(df_upload.head())
        
        col_options = list(df_upload.columns)
        chosen_col = st.selectbox("Pilih kolom yang berisi review", col_options)
        n_preview = st.number_input("Jumlah baris preview", min_value=1, max_value=500, value=10)
        
        if st.button("▶ Jalankan prediksi batch"):
            if model is None or tfidf is None:
                st.error("❌ Model / TF-IDF belum tersedia.")
            else:
                texts = df_upload[chosen_col].fillna("").astype(str).tolist()
                cleaned_texts = [clean_text(t) for t in texts]
                try:
                    X_vec = tfidf.transform(cleaned_texts)
                except Exception as e:
                    st.error(f"❌ Error saat transform batch: {e}")
                    st.stop()
                    
                probs, preds, classes = get_proba_and_pred(model, X_vec)
                proba_df = pd.DataFrame(probs, columns=[f"prob_{c}" for c in classes])
                out = df_upload.copy().reset_index(drop=True)
                out["pred_sentiment"] = preds
                out = pd.concat([out, proba_df], axis=1)
                
                st.success("✅ Selesai! Preview hasil:")
                st.dataframe(out.head(n_preview))
                
                csv_bytes = out.to_csv(index=False).encode("utf-8")
                st.download_button("⬇ Download hasil (CSV)", csv_bytes, "prediksi_hasil.csv", "text/csv")
    except Exception as e:
        st.error(f"❌ Gagal membaca CSV: {e}")

st.markdown("---")

# ===========================
# EDA SECTION
# ===========================
st.markdown("## 📈 Exploratory Data Analysis (Optional)")
with st.expander("Upload sample CSV untuk EDA"):
    sample_file = st.file_uploader("Upload sample dataset", type=["csv"], key="eda")
    if sample_file is not None:
        try:
            df_s = pd.read_csv(sample_file)
            st.write("*Preview:*")
            st.dataframe(df_s.head())
            
            possible = [c for c in df_s.columns if any(k in c.lower() for k in ["review","text","comment"])]
            if len(possible) == 0:
                chosen = st.selectbox("Pilih kolom teks", df_s.columns)
            else:
                chosen = st.selectbox("Pilih kolom teks", possible)
                
            if model is not None and tfidf is not None:
                texts = df_s[chosen].fillna("").astype(str).tolist()
                cleaned_texts = [clean_text(t) for t in texts]
                X_vec = tfidf.transform(cleaned_texts)
                probs, preds, classes = get_proba_and_pred(model, X_vec)
                df_s["pred_sentiment"] = preds
                st.write("*Distribusi prediksi:*")
                st.bar_chart(df_s["pred_sentiment"].value_counts())
                
            text_all = " ".join(df_s[chosen].fillna("").astype(str).tolist())
            wc = WordCloud(width=800, height=400, background_color="white").generate(text_all)
            fig, ax = plt.subplots(figsize=(10,4))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"❌ Error EDA: {e}")

if st.button("🌓", key="hidden_theme_toggle", help="Toggle Theme"):
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
    st.rerun()

st.markdown("---")
st.caption("📌 Aplikasi ini untuk demo final project. Pastikan model.pkl & tfidf.pkl cocok (dilatih dengan TF-IDF yang sama).")\
