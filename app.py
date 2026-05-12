import streamlit as st
import requests
import pandas as pd

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0E1117;
        color: white;
    }

    textarea {
        font-size: 16px !important;
    }

    .big-font {
        font-size: 18px !important;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("📌 Model Info")

st.sidebar.write("Model Versions:")
st.sidebar.write("• BiLSTM v2")
st.sidebar.write("• DistilBERT")

st.sidebar.write("---")

st.sidebar.write("Dataset:")
st.sidebar.write("• ISOT Dataset")
st.sidebar.write("• Real-world samples")
st.sidebar.write("• Twitter noisy samples")
st.sidebar.write("• Blog clickbait samples")

st.sidebar.write("---")

st.sidebar.write("Tech Stack:")
st.sidebar.write("• FastAPI")
st.sidebar.write("• Streamlit")
st.sidebar.write("• TensorFlow")
st.sidebar.write("• Transformers")

# ==========================================
# API STATUS CHECK
# ==========================================
try:
    health = requests.get("http://127.0.0.1:8000/health")

    if health.status_code == 200:
        st.sidebar.success("🟢 API Connected")
    else:
        st.sidebar.error("🔴 API Offline")

except:
    st.sidebar.error("🔴 API Offline")

# ==========================================
# TITLE
# ==========================================
st.title("📰 Fake News Detection System")

st.markdown("""
Detect whether a news article is **Real** or **Fake**
using Deep Learning models.
""")

# ==========================================
# SESSION STATE
# ==========================================
if "history" not in st.session_state:
    st.session_state.history = []

# ==========================================
# EXAMPLE BUTTONS
# ==========================================
col1, col2 = st.columns(2)

with col1:
    if st.button("Load Real Example"):

        st.session_state["news_text"] = """
        Reuters reported that global oil prices increased on Tuesday
        as investors reacted to supply concerns and ongoing geopolitical tensions.
        Analysts expect market volatility to continue throughout the quarter.
        """

with col2:
    if st.button("Load Fake Example"):

        st.session_state["news_text"] = """
        Scientists secretly discovered a miracle fruit that can cure
        every disease instantly but governments are hiding the truth
        from the public.
        """

# ==========================================
# MODEL SELECTION
# ==========================================
model_type = st.selectbox(
    "Choose Model",
    ["bilstm", "bert"]
)

# ==========================================
# TEXT INPUT
# ==========================================
text_input = st.text_area(
    "Enter News Text",
    value=st.session_state.get("news_text", ""),
    height=250,
    placeholder="Paste news article here..."
)

# ==========================================
# CHARACTER COUNT
# ==========================================
st.write(f"Characters: {len(text_input)}")

# ==========================================
# PREDICT BUTTON
# ==========================================
if st.button("Predict"):

    if text_input.strip() == "":
        st.warning("⚠️ Please enter news text.")

    else:

        payload = {
            "text": text_input,
            "model_type": model_type
        }

        try:

            # ==================================
            # LOADING SPINNER
            # ==================================
            with st.spinner("Analyzing news article..."):

                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                    json=payload
                )

                result = response.json()

            prediction = result["prediction"]
            confidence = result["confidence"]
            latency = result["latency"]

            # ==================================
            # STORE HISTORY
            # ==================================
            st.session_state.history.append({
                "prediction": prediction,
                "confidence": confidence,
                "model": model_type
            })

            # ==================================
            # RESULT DISPLAY
            # ==================================
            st.subheader("📊 Prediction Result")

            if prediction == "Real":

                st.success("✅ REAL NEWS")

            else:

                st.error("🚨 FAKE NEWS")

            # ==================================
            # METRICS
            # ==================================
            st.markdown(f"### Confidence: {confidence}")

            # Progress Bar
            st.progress(int(confidence * 100))

            st.write(f"**Model:** {model_type}")
            st.write(f"**Latency:** {latency} sec")

        except Exception as e:

            st.error(f"❌ Error connecting to API: {e}")

# ==========================================
# PREDICTION HISTORY
# ==========================================
if len(st.session_state.history) > 0:

    st.subheader("🕘 Prediction History")

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(
        history_df,
        use_container_width=True
    )

# ==========================================
# MODEL LIMITATIONS
# ==========================================
st.warning("""
⚠️ Model Limitations

This model may struggle with:
- satire articles
- political bias
- highly opinionated content
- unseen writing styles
- extremely short text
""")

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")

st.markdown(
    """
    <center>
    Built with ❤️ using FastAPI + Streamlit + TensorFlow
    </center>
    """,
    unsafe_allow_html=True
)