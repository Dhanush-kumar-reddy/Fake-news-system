import streamlit as st
import requests

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

.stTextArea textarea {
    font-size: 16px;
}

.result-box {
    padding: 1rem;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# RENDER API URL
# =====================================================
API_URL = "https://fake-news-system-3wcv.onrender.com/predict"

HEALTH_URL = "https://fake-news-system-3wcv.onrender.com/health"

# =====================================================
# TITLE
# =====================================================
st.title("📰 Fake News Detection System")

st.markdown("""
Detect whether a news article is **Real** or **Fake**
using a Deep Learning BiLSTM model.
""")

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("📌 Project Info")

st.sidebar.write("### Model")
st.sidebar.write("BiLSTM v2")

st.sidebar.write("### Dataset")
st.sidebar.write("- ISOT Dataset")
st.sidebar.write("- Real-world samples")
st.sidebar.write("- Blog clickbait samples")
st.sidebar.write("- Twitter noisy samples")

st.sidebar.write("### Stack")
st.sidebar.write("- FastAPI")
st.sidebar.write("- Streamlit")
st.sidebar.write("- TensorFlow")

# =====================================================
# API HEALTH CHECK
# =====================================================
try:

    health = requests.get(HEALTH_URL)

    if health.status_code == 200:
        st.sidebar.success("🟢 API Connected")
    else:
        st.sidebar.error("🔴 API Offline")

except:
    st.sidebar.error("🔴 API Offline")

# =====================================================
# EXAMPLE BUTTONS
# =====================================================
col1, col2 = st.columns(2)

with col1:

    if st.button("Load Real Example"):

        st.session_state.news_text = """
        Reuters reported that global markets showed positive growth
        after central bank officials signaled stable interest rates
        and improved economic outlook.
        """

with col2:

    if st.button("Load Fake Example"):

        st.session_state.news_text = """
        Scientists secretly discovered a hidden fruit that can cure
        every disease instantly but governments are hiding the truth.
        """

# =====================================================
# TEXT INPUT
# =====================================================
text_input = st.text_area(
    "Enter News Text",
    value=st.session_state.get("news_text", ""),
    height=250,
    placeholder="Paste news article here..."
)

# =====================================================
# CHARACTER COUNT
# =====================================================
st.caption(f"Characters: {len(text_input)}")

# =====================================================
# PREDICT BUTTON
# =====================================================
if st.button("Predict"):

    if text_input.strip() == "":

        st.warning("⚠️ Please enter some news text.")

    else:

        payload = {
            "text": text_input
        }

        try:

            with st.spinner("Analyzing article..."):

                response = requests.post(
                    API_URL,
                    json=payload,
                    timeout=60
                )

                result = response.json()

            prediction = result["prediction"]
            confidence = result["confidence"]
            latency = result["latency"]

            # =========================================
            # RESULT
            # =========================================
            st.subheader("📊 Prediction Result")

            if prediction == "Real":

                st.success("✅ REAL NEWS")

            else:

                st.error("🚨 FAKE NEWS")

            # =========================================
            # CONFIDENCE
            # =========================================
            st.write(f"### Confidence: {confidence}")

            st.progress(int(confidence * 100))

            # =========================================
            # DETAILS
            # =========================================
            st.write(f"**Latency:** {latency} sec")
            st.write("**Model:** BiLSTM v2")

        except Exception as e:

            st.error("❌ Error connecting to API")

            st.exception(e)

# =====================================================
# LIMITATIONS
# =====================================================
st.warning("""
⚠️ Model Limitations

This model may struggle with:
- satire articles
- highly opinionated text
- very short content
- unseen writing styles
""")

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")

st.markdown("""
<center>
Built using FastAPI + Streamlit + TensorFlow
</center>
""", unsafe_allow_html=True)