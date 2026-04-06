# ==========================================================
# 🧠 Emotion AI Pro — NeuroVision (Enterprise + Auto Language)
# 🎨 Radical UI + Stable Fixes + AR/EN Support
# ==========================================================

# ==============================
# 📦 Imports
# ==============================
import streamlit as st
import requests
import os
from dotenv import load_dotenv
from PIL import Image
import plotly.graph_objects as go
import assemblyai as aai
from openai import OpenAI
from io import BytesIO
import tempfile
from langdetect import detect

# ==============================
# 📦 NEW (Tone Analysis)
# ==============================
import librosa
import numpy as np

# ==========================================================
# 🔐 API Keys
# ==========================================================
load_dotenv()

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

missing_keys = []

if not HUGGINGFACE_API_KEY:
    missing_keys.append("HUGGINGFACE_API_KEY")

if not ASSEMBLYAI_API_KEY:
    missing_keys.append("ASSEMBLYAI_API_KEY")

if not OPENAI_API_KEY:
    missing_keys.append("OPENAI_API_KEY")

if missing_keys:
    st.error(f"❌ المفاتيح الناقصة: {', '.join(missing_keys)}")
    st.info("💡 تأكد من إضافتها في Environment Variables في منصة النشر")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ==========================================================
# 🎨 Page Config
# ==========================================================
st.set_page_config(
    page_title="Emotion AI Pro — NeuroVision",
    page_icon="🧠",
    layout="wide"
)

# ==========================================================
# 🌌 UI Design
# ==========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    direction: rtl;
    font-family: 'Tajawal', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 20% 30%, #1a1a2e, #0f3460, #16213e);
    background-size: 200% 200%;
    animation: moveBG 12s ease infinite;
}

@keyframes moveBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

.neo-card {
    background: rgba(255,255,255,0.05);
    border-radius: 25px;
    padding: 30px;
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 8px 40px rgba(0,0,0,0.4);
    transition: 0.4s ease;
}

.neo-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 20px 60px rgba(0,0,0,0.6);
}

.stButton>button {
    background: linear-gradient(135deg,#00c6ff,#0072ff);
    color:white;
    border:none;
    padding:14px 35px;
    border-radius:50px;
    font-size:18px;
    font-weight:600;
    transition:0.3s;
}

.stButton>button:hover {
    transform:scale(1.07);
    box-shadow:0 0 25px #00c6ff;
}

h1 {
    text-align:center;
    font-size:40px;
    font-weight:800;
    margin-bottom:30px;
}

.plotly-graph-div {
    border-radius:20px !important;
    overflow:hidden;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# 🧠 HEADER
# ==========================================================
st.title("🧠 Emotion AI Pro — NeuroVision")

# ==========================================================
# 📥 INPUT SECTION
# ==========================================================
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='neo-card'>", unsafe_allow_html=True)
    st.subheader("📷 تحليل تعابير الوجه")

    image_option = st.radio(
        "اختر مصدر الصورة:",
        ["رفع صورة", "التقاط من الكاميرا"],
        horizontal=True
    )

    image_bytes = None

    if image_option == "رفع صورة":
        uploaded_img = st.file_uploader("اختر صورة", type=["jpg", "png"])
        if uploaded_img:
            image_bytes = uploaded_img.getvalue()
    else:
        camera_img = st.camera_input("التقط صورة")
        if camera_img:
            image_bytes = camera_img.getvalue()

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='neo-card'>", unsafe_allow_html=True)
    st.subheader("🎤 تحليل نبرة الصوت")

    audio_option = st.radio(
        "اختر مصدر الصوت:",
        ["رفع ملف صوتي", "تسجيل من الميكروفون"],
        horizontal=True
    )

    audio_bytes = None

    if audio_option == "رفع ملف صوتي":
        uploaded_audio = st.file_uploader("اختر صوت", type=["mp3", "wav", "m4a"])
        if uploaded_audio:
            audio_bytes = uploaded_audio.getvalue()
    else:
        recorded_audio = st.audio_input("سجل صوتك")
        if recorded_audio:
            audio_bytes = recorded_audio.getvalue()

    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================================
# 📷 IMAGE ANALYSIS (كما هو)
# ==========================================================
def analyze_image(image_bytes):

    try:
        API_URL = "https://router.huggingface.co/hf-inference/models/trpakov/vit-face-expression"

        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
            "Content-Type": "application/octet-stream"
        }

        response = requests.post(API_URL, headers=headers, data=image_bytes, timeout=60)
        result = response.json()

        dominant = max(result, key=lambda x: x['score'])
        return result, dominant

    except Exception as e:
        st.error(f"❌ خطأ: {e}")
        return None, None

# ==========================================================
# 🎤 AUDIO ANALYSIS (🔥 التغيير هنا فقط)
# ==========================================================
def analyze_audio(audio_bytes):

    try:

        # 1️⃣ تحويل الصوت إلى نص
        aai.settings.api_key = ASSEMBLYAI_API_KEY

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        config = aai.TranscriptionConfig(
            speech_models=["universal"]
        )

        transcript = aai.Transcriber().transcribe(tmp_path, config=config)

        if transcript.status == aai.TranscriptStatus.error:
            st.error(f"❌ فشل التفريغ: {transcript.error}")
            return None, None

        text = transcript.text or "لا يوجد نص."

        # 2️⃣ تحليل نبرة الصوت
        y, sr = librosa.load(tmp_path)

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        energy = np.mean(librosa.feature.rms(y=y))

        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch = np.mean(pitches[magnitudes > np.median(magnitudes)])

        os.remove(tmp_path)

        # 3️⃣ تصنيف النبرة
        tone = ""

        tone += "سريع، " if tempo > 120 else "بطيء، "
        tone += "طاقة عالية، " if energy > 0.02 else "طاقة منخفضة، "
        tone += "نبرة مرتفعة (توتر/فرح)" if pitch > 150 else "نبرة منخفضة (هدوء/حزن)"

        return text, tone

    except Exception as e:
        st.error(f"❌ خطأ في الصوت: {e}")
        return None, None

# ==========================================================
# 🌍 AI DIAGNOSIS (تم تعديل بسيط فقط)
# ==========================================================
def generate_diagnosis(image_emotion, audio_text):

    try:
        try:
            detected_lang = detect(audio_text)
        except:
            detected_lang = "ar"

        prompt = f"""
Emotion: {image_emotion}

Speech:
{audio_text}
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        return response.choices[0].message.content, detected_lang

    except Exception as e:
        st.error(f"❌ خطأ في OpenAI: {e}")
        return None, None

# ==========================================================
# 🚀 START
# ==========================================================
if st.button("🚀 بدء التحليل الذكي"):

    if image_bytes and audio_bytes:

        emotions, dominant = analyze_image(image_bytes)
        text, tone = analyze_audio(audio_bytes)

        diagnosis, lang = generate_diagnosis(
            dominant["label"],
            text + "\nTone: " + tone
        )

        st.image(Image.open(BytesIO(image_bytes)))
        st.write("🎤 النبرة:", tone)
        st.write("🧠 التحليل:", diagnosis)

    else:
        st.warning("⚠ يرجى إدخال صورة وصوت أولاً.")