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

st.set_page_config(page_title="Amazon Review Sentiment", layout="wide", page_icon="🛒")

# ============================
# THEME SYSTEM
# ============================
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# ===========================
# THEME CSS
# ===========================
if st.session_state.theme == "light":
    st.markdown("""
    <style>
    /* MAIN */
    [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        background-color: #ffffff !important;
    }
    
    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background-color: #f5f5f5 !important;
    }
    [data-testid="stSidebar"] * {
        color: #000000 !important;
    }
    
    /* CONTENT */
    .block-container {
        background-color: #ffffff !important;
    }
    
    /* TEXT */
    .stMarkdown, .stMarkdown *, label, p, h1, h2, h3, h4, h5, h6, span, div {
        color: #000000 !important;
    }
    
    /* DATAFRAME */
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
    
    /* BUTTONS */
    .stButton > button {
        background-color: #ffd700 !important;
        color: #000000 !important;
        border: 1px solid #cccccc !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover {
        background-color: #ffed4e !important;
        border: 1px solid #999999 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
    }
    
    /* INPUTS */
    textarea, input, .stTextArea textarea, .stTextInput input {
        background-color: #f9f9f9 !important;
        color: #000000 !important;
        border: 1px solid #cccccc !important;
    }
    
    /* EXPANDER */
    [data-testid="stExpander"] {
        background-color: #f9f9f9 !important;
        border: 1px solid #e0e0e0 !important;
    }
    [data-testid="stExpander"] * {
        color: #000000 !important;
    }
    
    /* FILE UPLOADER */
    [data-testid="stFileUploader"] {
        background-color: #f9f9f9 !important;
    }
    
    /* METRICS */
    [data-testid="stMetricValue"] {
        color: #000000 !important;
    }
    
    /* HIDE THEME BUTTON */
    button[key="hidden_theme_toggle"] {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

else:  # dark theme
    st.markdown("""
    <style>
    /* MAIN */
    [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        background-color: #0e1117 !important;
    }
    
    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background-color: #1a1d24 !important;
    }
    [data-testid="stSidebar"] * {
        color: #fafafa !important;
    }
    
    /* CONTENT */
    .block-container {
        background-color: #0e1117 !important;
    }
    
    /* TEXT */
    .stMarkdown, .stMarkdown *, label, p, h1, h2, h3, h4, h5, h6, span, div {
        color: #fafafa !important;
    }
    
    /* DATAFRAME */
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
    
    /* BUTTONS */
    .stButton > button {
        background-color: #262a33 !important;
        color: #fafafa !important;
        border: 1px solid #3d4450 !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover {
        background-color: #2d323d !important;
        border: 1px solid #4d5562 !important;
        box-shadow: 0 2px 8px rgba(255,255,255,0.1) !important;
    }
    
    /* INPUTS */
    textarea, input, .stTextArea textarea, .stTextInput input {
        background-color: #1a1d24 !important;
        color: #fafafa !important;
        border: 1px solid #3d4450 !important;
    }
    
    /* EXPANDER */
    [data-testid="stExpander"] {
        background-color: #1a1d24 !important;
        border: 1px solid #3d4450 !important;
    }
    [data-testid="stExpander"] * {
        color: #fafafa !important;
    }
    
    /* FILE UPLOADER */
    [data-testid="stFileUploader"] {
        background-color: #1a1d24 !important;
    }
    
    /* METRICS */
    [data-testid="stMetricValue"] {
        color: #fafafa !important;
    }
    
    /* HIDE THEME BUTTON */
    button[key="hidden_theme_toggle"] {
        display: none !important;
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
    """Clean and preprocess text"""
    if not isinstance(t, str):
        return ""
    t = t.lower()
    t = re.sub(r"[^a-zA-Z\s]", " ", t)
    tokens = [w for w in t.split() if (w not in STOPWORDS) and (len(w) > 1)]
    return " ".join(tokens)

def get_proba_and_pred(model, X_vector):
    """Get prediction probabilities from model"""
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
    """Load pickle file from bytes"""
    return pickle.loads(path_bytes.read())

# ===========================
# SIDEBAR
# ===========================
with st.sidebar:
    st.markdown("### 🎨 Tema Aplikasi")
    
    # Beautiful Theme Toggle
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
        justify-content: center;
        padding: 10px 0;
        background: transparent;
    }}
    
    .toggle-wrapper {{
        position: relative;
        width: 220px;
        height: 55px;
        background: linear-gradient(90deg, #ff6b9d 0%, #ffa06b 50%, #ffd93d 100%);
        border-radius: 30px;
        cursor: pointer;
        box-shadow: 0 6px 20px rgba(255, 107, 157, 0.4);
        transition: all 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        overflow: hidden;
        user-select: none;
    }}
    
    .toggle-wrapper.night-mode {{
        background: linear-gradient(90deg, #2a5298 0%, #1e3c72 50%, #4facfe 100%);
        box-shadow: 0 6px 20px rgba(79, 172, 254, 0.4);
    }}
    
    .toggle-slider {{
        position: absolute;
        top: 4px;
        left: 4px;
        width: 100px;
        height: 47px;
        background: white;
        border-radius: 30px;
        transition: all 0.5s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        box-shadow: 0 3px 12px rgba(0,0,0,0.25);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 12px;
        color: #ff6b9d;
        letter-spacing: 0.5px;
        z-index: 2;
    }}
    
    .toggle-wrapper.night-mode .toggle-slider {{
        left: 116px;
        color: #2a5298;
    }}
    
    .toggle-option {{
        position: absolute;
        top: 50%;
        transform: translateY(-50%);
        display: flex;
        align-items: center;
        gap: 6px;
        font-weight: 600;
        font-size: 11px;
        color: white;
        transition: all 0.3s ease;
        z-index: 1;
        pointer-events: none;
    }}
    
    .toggle-option-left {{
        left: 18px;
    }}
    
    .toggle-option-right {{
        right: 18px;
    }}
    
    .toggle-wrapper.night-mode .toggle-option-left {{
        opacity: 0.6;
    }}
    
    .toggle-wrapper:not(.night-mode) .toggle-option-right {{
        opacity: 0.6;
    }}
    
    .toggle-icon {{
        font-size: 20px;
        line-height: 1;
    }}
    
    .sun-icon {{
        animation: rotate 20s linear infinite;
        filter: drop-shadow(0 0 4px rgba(255, 255, 255, 0.8));
    }}
    
    .moon-icon {{
        animation: pulse 3s ease-in-out infinite;
        filter: drop-shadow(0 0 4px rgba(255, 255, 255, 0.8));
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
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }}
    
    .toggle-wrapper:active {{
        transform: scale(0.97);
    }}
    </style>
    </head>
    <body>
    <div class="theme-toggle-container">
        <div class="toggle-wrapper {'night-mode' if st.session_state.theme == 'dark' else ''}" id="toggleWrapper" onclick="toggleTheme()">
            <div class="toggle-slider" id="toggleSlider">
                <span id="sliderText">{'NIGHT MODE' if st.session_state.theme == 'dark' else 'DAY MODE'}</span>
            </div>
            <div class="toggle-option toggle-option-left">
                <span class="toggle-icon sun-icon">☀️</span>
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
        
        // Toggle visual
        if (wrapper.classList.contains('night-mode')) {{
            wrapper.classList.remove('night-mode');
            sliderText.textContent = 'DAY MODE';
        }} else {{
            wrapper.classList.add('night-mode');
            sliderText.textContent = 'NIGHT MODE';
        }}
        
        // Trigger Streamlit button
        setTimeout(() => {{
            try {{
                const parentDoc = window.parent.document;
                const allButtons = parentDoc.querySelectorAll('button');
                for (let btn of allButtons) {{
                    const btnText = btn.innerText || btn.textContent || '';
                    if (btnText.includes('🌓')) {{
                        btn.click();
                        break;
                    }}
                }}
            }} catch(e) {{
                console.error('Theme toggle error:', e);
            }}
        }}, 50);
    }}
    </script>
    </body>
    </html>
    """
    components.html(toggle_html, height=85, scrolling=False)
    
    # Hidden button for theme switching
    if st.button("🌓", key="hidden_theme_toggle"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()
    
    st.markdown("---")
    
    # Model Upload Section
    with st.expander("📦 Model / Vectorizer", expanded=False):
        st.write("Upload model dan TF-IDF vectorizer:")
        uploaded_model = st.file_uploader("Upload model.pkl", type=["pkl","pickle"], key="m1")
        uploaded_tfidf = st.file_uploader("Upload tfidf.pkl", type=["pkl","pickle"], key="v1")
        use_local = st.checkbox("Gunakan file lokal", value=True, help="Gunakan model.pkl & tfidf.pkl dari folder aplikasi")

# ===========================
# LOAD MODELS
# ===========================
model = None
tfidf = None

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
        st.sidebar.success("✅ Model berhasil dimuat")
    except Exception as e:
        st.sidebar.error(f"❌ Gagal load model: {e}")

if uploaded_tfidf is not None:
    try:
        tfidf = load_pickle(uploaded_tfidf)
        st.sidebar.success("✅ TF-IDF berhasil dimuat")
    except Exception as e:
        st.sidebar.error(f"❌ Gagal load tfidf: {e}")

# ===========================
# MAIN HEADER
# ===========================
col1, col2 = st.columns([4,1])
with col1:
    st.markdown('<h1>🛒 Amazon Review Sentiment Analysis</h1>', unsafe_allow_html=True)
    st.markdown("Analisis sentimen review produk Amazon menggunakan Machine Learning")

with col2:
    st.image("https://www.freeiconspng.com/uploads/amazon-icon-6.png", width=100)

st.markdown("---")

# Model Status
if model is None or tfidf is None:
    st.warning("⚠️ Model atau TF-IDF belum tersedia. Silakan upload atau pastikan file tersedia di folder aplikasi.")
    st.info("💡 **Cara menggunakan:** Upload file `model.pkl` dan `tfidf.pkl` melalui sidebar, atau letakkan di folder aplikasi.")
    st.stop()
else:
    st.success("✅ Model dan TF-IDF siap digunakan!")

# ===========================
# SINGLE PREDICTION
# ===========================
st.markdown("### 📝 Prediksi Review Tunggal")

if "review_box" not in st.session_state:
    st.session_state["review_box"] = ""

colA, colB = st.columns([4,1])

with colA:
    text_input = st.text_area(
        "Masukkan review di sini:",
        height=140,
        key="review_box",
        placeholder="Contoh: The product stopped working after 2 days. Very disappointed with the quality."
    )

with colB:
    st.write("**Aksi Cepat:**")

    def reset_input():
        st.session_state["review_box"] = ""

    st.button("🔄 Reset", on_click=reset_input, use_container_width=True)

    sample_examples = {
        "positive": [
            "Amazing product! Exceeded my expectations.",
            "Works perfectly, very satisfied!",
            "High quality and fast delivery.",
            "Best purchase ever! Highly recommended."
        ],
        "neutral": [
            "The product is okay, nothing special.",
            "Average experience, not bad not great.",
            "It's fine for the price.",
            "Works as described, no complaints."
        ],
        "negative": [
            "Terrible quality. Very disappointed.",
            "Stopped working after two days.",
            "Not worth the money at all.",
            "Poor customer service and defective product."
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

if st.button("🔍 Prediksi Sentimen", use_container_width=True, type="primary"):
    if not isinstance(text_input, str) or text_input.strip() == "":
        st.error("❌ Masukkan teks review terlebih dahulu.")
    else:
        with st.spinner("Menganalisis sentimen..."):
            cleaned = clean_text(text_input)
            try:
                vec = tfidf.transform([cleaned])
                probs, preds, classes = get_proba_and_pred(model, vec)
                pred = preds[0]
                proba_map = {classes[i]: float(probs[0,i]) for i in range(len(classes))}
                
                # Display results
                col_r1, col_r2 = st.columns([1, 2])
                
                with col_r1:
                    sentiment_emoji = {
                        "positive": "😊",
                        "neutral": "😐",
                        "negative": "😞"
                    }
                    st.metric("Prediksi Sentimen", 
                             f"{sentiment_emoji.get(pred.lower(), '📊')} {pred.upper()}")
                
                with col_r2:
                    st.markdown("**Confidence Score:**")
                    dfc = pd.DataFrame.from_dict(proba_map, orient="index", columns=["Probability"]).sort_values("Probability", ascending=False)
                    dfc["Percentage"] = (dfc["Probability"] * 100).round(2).astype(str) + "%"
                    st.dataframe(dfc[["Percentage"]], use_container_width=True)
                
                st.bar_chart(dfc["Probability"])
                
            except Exception as e:
                st.error(f"❌ Error saat prediksi: {e}")

st.markdown("---")

# ===========================
# BATCH PREDICTION
# ===========================
st.markdown("## 📊 Prediksi Batch (Upload CSV)")
st.write("Upload file CSV yang berisi kolom teks review untuk analisis massal.")

uploaded_csv = st.file_uploader("📂 Pilih file CSV", type=["csv"], help="Format CSV dengan header kolom")

if uploaded_csv is not None:
    try:
        df_upload = pd.read_csv(uploaded_csv)
        st.success(f"✅ File berhasil dimuat: {len(df_upload)} baris")
        
        st.write("**Preview Data:**")
        st.dataframe(df_upload.head(10), use_container_width=True)
        
        col_options = list(df_upload.columns)
        chosen_col = st.selectbox("📋 Pilih kolom yang berisi review:", col_options)
        n_preview = st.number_input("Jumlah baris preview hasil:", min_value=1, max_value=500, value=10)
        
        if st.button("▶️ Jalankan Prediksi Batch", type="primary"):
            with st.spinner(f"Memproses {len(df_upload)} review..."):
                texts = df_upload[chosen_col].fillna("").astype(str).tolist()
                cleaned_texts = [clean_text(t) for t in texts]
                
                try:
                    X_vec = tfidf.transform(cleaned_texts)
                    probs, preds, classes = get_proba_and_pred(model, X_vec)
                    proba_df = pd.DataFrame(probs, columns=[f"prob_{c}" for c in classes])
                    
                    out = df_upload.copy().reset_index(drop=True)
                    out["predicted_sentiment"] = preds
                    out = pd.concat([out, proba_df], axis=1)
                    
                    st.success(f"✅ Prediksi selesai untuk {len(out)} review!")
                    
                    # Statistics
                    col_s1, col_s2, col_s3 = st.columns(3)
                    sentiment_counts = out["predicted_sentiment"].value_counts()
                    
                    with col_s1:
                        st.metric("😊 Positive", sentiment_counts.get("positive", 0))
                    with col_s2:
                        st.metric("😐 Neutral", sentiment_counts.get("neutral", 0))
                    with col_s3:
                        st.metric("😞 Negative", sentiment_counts.get("negative", 0))
                    
                    st.write("**Distribusi Sentimen:**")
                    st.bar_chart(sentiment_counts)
                    
                    st.write(f"**Preview Hasil (Top {n_preview} baris):**")
                    st.dataframe(out.head(n_preview), use_container_width=True)
                    
                    # Download button
                    csv_bytes = out.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download Hasil Lengkap (CSV)", 
                        csv_bytes, 
                        "sentiment_analysis_results.csv", 
                        "text/csv",
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error saat batch prediction: {e}")
                    
    except Exception as e:
        st.error(f"❌ Gagal membaca file CSV: {e}")

st.markdown("---")

# ===========================
# EDA SECTION
# ===========================
st.markdown("## 📈 Exploratory Data Analysis")
st.write("Analisis dan visualisasi dataset untuk insight lebih dalam.")

with st.expander("🔍 Buka EDA Tools"):
    sample_file = st.file_uploader("Upload dataset untuk analisis", type=["csv"], key="eda")
    
    if sample_file is not None:
        try:
            df_s = pd.read_csv(sample_file)
            
            tab1, tab2, tab3 = st.tabs(["📋 Data Overview", "📊 Sentiment Analysis", "☁️ Word Cloud"])
            
            with tab1:
                st.write("**Dataset Information:**")
                col_info1, col_info2, col_info3 = st.columns(3)
                with col_info1:
                    st.metric("Total Rows", len(df_s))
                with col_info2:
                    st.metric("Total Columns", len(df_s.columns))
                with col_info3:
                    st.metric("Missing Values", df_s.isnull().sum().sum())
                
                st.write("**Data Preview:**")
                st.dataframe(df_s.head(20), use_container_width=True)
                
                st.write("**Column Statistics:**")
                st.dataframe(df_s.describe(), use_container_width=True)
            
            with tab2:
                possible = [c for c in df_s.columns if any(k in c.lower() for k in ["review","text","comment"])]
                if len(possible) == 0:
                    chosen = st.selectbox("Pilih kolom teks untuk analisis:", df_s.columns)
                else:
                    chosen = st.selectbox("Pilih kolom teks untuk analisis:", possible)
                
                if st.button("🔄 Analyze Sentiment Distribution"):
                    with st.spinner("Menganalisis sentimen..."):
                        texts = df_s[chosen].fillna("").astype(str).tolist()
                        cleaned_texts = [clean_text(t) for t in texts]
                        X_vec = tfidf.transform(cleaned_texts)
                        probs, preds, classes = get_proba_and_pred(model, X_vec)
                        df_s["predicted_sentiment"] = preds
                        
                        st.write("**Distribusi Sentimen:**")
                        sentiment_counts = df_s["predicted_sentiment"].value_counts()
                        
                        col_chart1, col_chart2 = st.columns(2)
                        with col_chart1:
                            st.bar_chart(sentiment_counts)
                        with col_chart2:
                            fig, ax = plt.subplots(figsize=(8, 6))
                            ax.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
                            ax.axis('equal')
                            st.pyplot(fig)
            
            with tab3:
                possible = [c for c in df_s.columns if any(k in c.lower() for k in ["review","text","comment"])]
                if len(possible) == 0:
                    chosen_wc = st.selectbox("Pilih kolom untuk Word Cloud:", df_s.columns, key="wc_col")
                else:
                    chosen_wc = st.selectbox("Pilih kolom untuk Word Cloud:", possible, key="wc_col")
                
                if st.button("☁️ Generate Word Cloud"):
                    with st.spinner("Generating word cloud..."):
                        text_all = " ".join(df_s[chosen_wc].fillna("").astype(str).tolist())
                        wc = WordCloud(
                            width=1200, 
                            height=600, 
                            background_color="white",
                            colormap="viridis",
                            max_words=200
                        ).generate(text_all)
                        
                        fig, ax = plt.subplots(figsize=(15, 8))
                        ax.imshow(wc, interpolation="bilinear")
                        ax.axis("off")
                        st.pyplot(fig)
                        
                        st.info("💡 Word Cloud menampilkan kata-kata yang paling sering muncul dalam review.")
                        
        except Exception as e:
            st.error(f"❌ Error dalam EDA: {e}")

st.markdown("---")

# ===========================
# FOOTER
# ===========================
st.markdown("""
<div style='text-align: center; padding: 20px; color: gray;'>
    <p>📌 <strong>Amazon Review Sentiment Analysis Tool</strong></p>
    <p>Aplikasi ini menggunakan Machine Learning untuk menganalisis sentimen review produk</p>
    <p>Pastikan model.pkl dan tfidf.pkl telah dilatih dengan dataset yang sesuai</p>
    <p style='font-size: 12px; margin-top: 10px;'>Developed with ❤️ using
