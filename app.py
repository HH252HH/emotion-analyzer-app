# ==========================================================
# 🧠 Emotion AI Pro — NeuroVision (Enterprise + Auto Language)
# 🎨 Radical UI + Stable Fixes + AR/EN Support
# ==========================================================

# ==============================
# 📦 Imports (UNCHANGED + langdetect)
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
    st.subheader("🎤 تحليل المشاعر الصوتية")

    audio_option = st.radio(
        "اختر مصدر الصوت:",
        ["رفع ملف صوتي", "تسجيل من الميكروفون"],
        horizontal=True
    )

    audio_bytes = None
    if audio_option == "رفع ملف صوتي":
        uploaded_audio = st.file_uploader(
            "اختر صوت",
            type=["mp3", "wav", "m4a"]
        )
        if uploaded_audio:
            audio_bytes = uploaded_audio.getvalue()
    else:
        recorded_audio = st.audio_input("سجل صوتك")
        if recorded_audio:
            audio_bytes = recorded_audio.getvalue()

    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================================
# 📷 IMAGE ANALYSIS
# ==========================================================
def analyze_image(image_bytes):
    try:
        API_URL = "https://router.huggingface.co/hf-inference/models/trpakov/vit-face-expression"
        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
            "Content-Type": "application/octet-stream"
        }
        response = requests.post(API_URL, headers=headers, data=image_bytes, timeout=60)
        response.raise_for_status()
        result = response.json()

        if not isinstance(result, list) or len(result) == 0:
            st.error("❌ لم يتم العثور على وجه.")
            return None, None

        dominant = max(result, key=lambda x: x['score'])
        return result, dominant

    except Exception as e:
        st.error(f"❌ خطأ: {e}")
        return None, None

# ==========================================================
# 🎤 AUDIO ANALYSIS
# ==========================================================
def analyze_audio(audio_bytes):
    try:
        aai.settings.api_key = ASSEMBLYAI_API_KEY

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        config = aai.TranscriptionConfig(
            speech_models=["universal"],
            sentiment_analysis=True
        )

        transcript = aai.Transcriber().transcribe(tmp_path, config=config)
        os.remove(tmp_path)

        if transcript.status == aai.TranscriptStatus.error:
            st.error(f"❌ فشل التفريغ: {transcript.error}")
            return None, None

        return transcript.text or "لا يوجد نص.", transcript.sentiment_analysis or []

    except Exception as e:
        st.error(f"❌ خطأ في الصوت: {e}")
        return None, None

# ==========================================================
# 🌍 FINAL EMOTIONAL DIAGNOSIS (UPDATED ONLY HERE)
# ==========================================================
def generate_diagnosis(image_emotion, audio_text):
    try:
        try:
            detected_lang = detect(audio_text) if audio_text.strip() else "ar"
        except:
            detected_lang = "ar"

        if detected_lang == "en":
            system_msg = "You are a professional emotional and psychological AI analyst."

            final_prompt = f"""
            Face emotion analysis: {image_emotion}
            Speech content: {audio_text}

            Provide a unified emotional diagnosis by combining facial expression and speech analysis.

            Your response must include:
            1) Clear explanation of the person's emotional state.
            2) Interpretation of psychological condition.
            3) If emotional state is negative → provide practical steps to improve mood and mental state.
            4) If emotional state is positive → provide advice to maintain and strengthen well-being.

            Response must be clear, supportive, and professional.
            """

        else:
            system_msg = "أنت خبير تحليل نفسي وعاطفي احترافي."

            final_prompt = f"""
            تحليل تعابير الوجه: {image_emotion}
            محتوى الكلام: {audio_text}

            قدم تشخيصًا عاطفيًا موحدًا يجمع بين تحليل الصورة والصوت.

            يجب أن يتضمن الرد:
            1) شرح واضح للحالة العاطفية للشخص.
            2) تفسير للحالة النفسية العامة.
            3) إذا كانت الحالة سلبية → قدم خطوات عملية لتحسين المزاج والحالة النفسية.
            4) إذا كانت الحالة إيجابية → قدم نصائح للحفاظ على التوازن النفسي وتعزيز الحالة الجيدة.

            اجعل الرد داعمًا وواضحًا واحترافيًا.
            """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": final_prompt}
            ],
            temperature=0.7,
            max_tokens=800
        )

        return response.choices[0].message.content, detected_lang

    except Exception as e:
        st.error(f"❌ خطأ في OpenAI: {e}")
        return None, None

# ==========================================================
# 🚀 START ANALYSIS
# ==========================================================
if st.button("🚀 بدء التحليل الذكي"):

    if image_bytes and audio_bytes:

        progress = st.progress(0)

        progress.progress(20)
        emotions, dominant = analyze_image(image_bytes)
        if not emotions:
            st.stop()

        progress.progress(50)
        text, sentiments = analyze_audio(audio_bytes)
        if text is None:
            st.stop()

        progress.progress(70)
        diagnosis, lang = generate_diagnosis(dominant["label"], text)

        progress.progress(100)

        st.markdown("<div class='neo-card'>", unsafe_allow_html=True)
        colA, colB = st.columns(2)

        with colA:
            st.image(Image.open(BytesIO(image_bytes)), use_container_width=True)

        with colB:
            labels = [e["label"] for e in emotions]
            scores = [e["score"] for e in emotions]

            fig = go.Figure([go.Bar(x=labels, y=scores)])
            fig.update_layout(
                title="تحليل تعابير الوجه",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='neo-card'>", unsafe_allow_html=True)
        st.subheader("📊 التشخيص النهائي")
        st.info(f"🌍 اللغة المكتشفة: {'العربية' if lang == 'ar' else 'English'}")
        st.write(diagnosis)
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.warning("⚠ يرجى إدخال صورة وصوت أولاً.")