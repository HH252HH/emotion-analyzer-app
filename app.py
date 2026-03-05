# ==========================================================
# 🧠 Emotion AI Pro — NeuroVision
# Stable High Accuracy Version
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

# ==========================================================
<<<<<<< HEAD
# 🔐 API KEYS
# ==========================================================

=======
# 🔐 API Keys
# ==========================================================
>>>>>>> 2c7b26a16d7467b3c6ab1898e413e9af823d0a08
load_dotenv()

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

<<<<<<< HEAD
missing = []
=======
missing_keys = []
>>>>>>> 2c7b26a16d7467b3c6ab1898e413e9af823d0a08

if not HUGGINGFACE_API_KEY:
    missing.append("HUGGINGFACE_API_KEY")

if not ASSEMBLYAI_API_KEY:
    missing.append("ASSEMBLYAI_API_KEY")

if not OPENAI_API_KEY:
    missing.append("OPENAI_API_KEY")

if missing:
    st.error(f"❌ المفاتيح الناقصة: {', '.join(missing)}")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ==========================================================
# 🎨 PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Emotion AI Pro — NeuroVision",
    page_icon="🧠",
    layout="wide"
)

# ==========================================================
<<<<<<< HEAD
=======
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
</style>
""", unsafe_allow_html=True)

# ==========================================================
>>>>>>> 2c7b26a16d7467b3c6ab1898e413e9af823d0a08
# 🧠 HEADER
# ==========================================================

st.title("🧠 Emotion AI Pro — NeuroVision")

# ==========================================================
# 📥 INPUTS
# ==========================================================

col1, col2 = st.columns(2)

with col1:

    st.subheader("📷 تحليل تعابير الوجه")

    image_option = st.radio(
        "اختر مصدر الصورة:",
        ["رفع صورة", "التقاط من الكاميرا"],
        horizontal=True
    )

    image_bytes = None

    if image_option == "رفع صورة":

        img = st.file_uploader("اختر صورة", type=["jpg", "png"])

        if img:
            image_bytes = img.getvalue()

    else:

        cam = st.camera_input("التقط صورة")

        if cam:
            image_bytes = cam.getvalue()


with col2:

    st.subheader("🎤 تحليل الصوت")

    audio_option = st.radio(
        "اختر مصدر الصوت:",
        ["رفع ملف صوتي", "تسجيل"],
        horizontal=True
    )

    audio_bytes = None

    if audio_option == "رفع ملف صوتي":

        audio = st.file_uploader(
            "اختر ملف صوت",
            type=["wav","mp3","m4a"]
        )

        if audio:
            audio_bytes = audio.getvalue()

    else:

        rec = st.audio_input("سجل صوتك")

        if rec:
            audio_bytes = rec.getvalue()

# ==========================================================
# 📷 IMAGE ANALYSIS (HIGH STABILITY)
# ==========================================================

def analyze_image(image_bytes):

    try:

        API_URL = "https://router.huggingface.co/hf-inference/models/trpakov/vit-face-expression"

        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
            "Content-Type": "application/octet-stream"
        }

        response = requests.post(
            API_URL,
            headers=headers,
            data=image_bytes,
            timeout=120
        )

        response.raise_for_status()

        result = response.json()

        if isinstance(result, list) and isinstance(result[0], list):
            result = result[0]

        if not result:
            st.error("❌ لم يتم اكتشاف وجه")
            return None, None

        result_sorted = sorted(
            result,
            key=lambda x: x["score"],
            reverse=True
        )

        dominant = result_sorted[0]

        return result_sorted, dominant

    except Exception as e:

        st.error(f"❌ خطأ تحليل الصورة: {e}")
        return None, None


# ==========================================================
# 🎤 AUDIO ANALYSIS (HIGH ACCURACY)
# ==========================================================

def analyze_audio(audio_bytes):

    try:

        aai.settings.api_key = ASSEMBLYAI_API_KEY

        with tempfile.NamedTemporaryFile(delete=False) as tmp:

            tmp.write(audio_bytes)
            tmp_path = tmp.name

        config = aai.TranscriptionConfig(
            sentiment_analysis=True,
            language_detection=True
        )

        transcriber = aai.Transcriber()

        transcript = transcriber.transcribe(
            tmp_path,
            config=config
        )

        os.remove(tmp_path)

        if transcript.status == aai.TranscriptStatus.error:

            st.error(transcript.error)
            return None, None

        text = transcript.text or ""

        sentiments = transcript.sentiment_analysis or []

        return text, sentiments

    except Exception as e:

        st.error(f"❌ خطأ تحليل الصوت: {e}")
        return None, None


# ==========================================================
<<<<<<< HEAD
# 🌍 AI DIAGNOSIS
=======
# 🌍 FINAL EMOTIONAL DIAGNOSIS (UPDATED ONLY HERE)
>>>>>>> 2c7b26a16d7467b3c6ab1898e413e9af823d0a08
# ==========================================================

def generate_diagnosis(image_emotion, audio_text):

    try:

        if not audio_text.strip():
            detected_lang = "ar"
        else:
            try:
                detected_lang = detect(audio_text[:200])
            except:
                detected_lang = "ar"

        if detected_lang == "en":
<<<<<<< HEAD

            system = """
You are an expert emotional intelligence AI psychologist.
Provide a professional emotional evaluation.
"""

            prompt = f"""
Facial Emotion:
{image_emotion}

Speech:
{audio_text}

Provide:

1 Emotional interpretation
2 Psychological indicators
3 Emotional state summary
4 Wellbeing recommendations
"""

        else:

            system = """
أنت خبير تحليل نفسي وعاطفي.
قدم تحليل احترافي للحالة.
"""

            prompt = f"""
العاطفة من الوجه:
{image_emotion}

النص من الصوت:
{audio_text}

قدم:

1 تفسير الحالة
2 احتمالية القلق
3 احتمالية التوتر
4 احتمالية الاكتئاب
5 توصيات عملية
"""
=======
            system_msg = "You are a professional emotional and psychological AI analyst."

            final_prompt = f"""
            Face emotion analysis: {image_emotion}
            Speech content: {audio_text}

            Provide a unified emotional diagnosis by combining facial expression and speech analysis.

            Your response must include clearly structured sections:

            1) Emotional State Classification:
               Explicitly state whether the emotional state is POSITIVE or NEGATIVE.

            2) Emotional Explanation:
               Explain the emotional and psychological condition clearly.

            3) Guidance:
               - If NEGATIVE → provide practical steps to improve mood and emotional stability.
               - If POSITIVE → provide advice to maintain and strengthen well-being.

            Make the response supportive, clear, and professional.
            """

        else:
            system_msg = "أنت خبير تحليل نفسي وعاطفي احترافي."

            final_prompt = f"""
            تحليل تعابير الوجه: {image_emotion}
            محتوى الكلام: {audio_text}

            قدم تشخيصًا عاطفيًا موحدًا يجمع بين تحليل الصورة والصوت.

            يجب أن يحتوي الرد على أقسام واضحة:

            1) تصنيف الحالة العاطفية:
               حدد بوضوح هل الحالة إيجابية أم سلبية.

            2) شرح الحالة:
               فسر الوضع العاطفي والنفسي بشكل مفهوم.

            3) التوصيات:
               - إذا كانت الحالة سلبية → قدم خطوات عملية لتحسين المزاج والاستقرار النفسي.
               - إذا كانت الحالة إيجابية → قدم نصائح للحفاظ على التوازن وتعزيز الحالة الجيدة.

            اجعل الرد داعمًا وواضحًا واحترافيًا.
            """
>>>>>>> 2c7b26a16d7467b3c6ab1898e413e9af823d0a08

        response = client.chat.completions.create(

            model="gpt-4o-mini",

            messages=[

                {"role": "system", "content": system},

                {"role": "user", "content": prompt}

            ],

            temperature=0.6,
            max_tokens=700
        )

        return response.choices[0].message.content, detected_lang

    except Exception as e:

        st.error(f"❌ خطأ OpenAI: {e}")
        return None, None


# ==========================================================
# 🚀 RUN ANALYSIS
# ==========================================================

if st.button("🚀 بدء التحليل"):

    if image_bytes and audio_bytes:

        progress = st.progress(0)

        progress.progress(25)

        emotions, dominant = analyze_image(image_bytes)

        if not emotions:
            st.stop()

        progress.progress(55)

        text, sentiments = analyze_audio(audio_bytes)

        if text is None:
            st.stop()

        progress.progress(80)

        diagnosis, lang = generate_diagnosis(
            dominant["label"],
            text
        )

        progress.progress(100)

<<<<<<< HEAD
=======
        st.markdown("<div class='neo-card'>", unsafe_allow_html=True)
>>>>>>> 2c7b26a16d7467b3c6ab1898e413e9af823d0a08
        colA, colB = st.columns(2)

        with colA:

            st.image(
                Image.open(BytesIO(image_bytes)),
                use_container_width=True
            )

        with colB:

            labels = [e["label"] for e in emotions]

            scores = [e["score"] for e in emotions]

            fig = go.Figure([
                go.Bar(x=labels, y=scores)
            ])

            fig.update_layout(
                title="تحليل تعابير الوجه",
                template="plotly_dark"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.subheader("📊 التشخيص")

        st.info(
            f"🌍 اللغة المكتشفة: {'العربية' if lang == 'ar' else 'English'}"
        )

        st.write(diagnosis)

        st.audio(audio_bytes)

    else:

        st.warning("⚠ يرجى إدخال صورة وصوت.")