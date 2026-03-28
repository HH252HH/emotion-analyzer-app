
# ==========================================================
# 🧠 Emotion AI Pro — NeuroVision (Ultra Dashboard Edition)
# 🎯 نسخة محسنة UI/UX احترافية
# ==========================================================

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
import librosa
import numpy as np

# ==========================================================
# 🔐 API Keys
# ==========================================================
load_dotenv()

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not all([HUGGINGFACE_API_KEY, ASSEMBLYAI_API_KEY, OPENAI_API_KEY]):
    st.error("❌ تأكد من جميع مفاتيح API")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ==========================================================
# 🎨 إعداد الصفحة + تصميم متطور
# ==========================================================
st.set_page_config(page_title="NeuroVision", layout="wide")

st.markdown("""
<style>
body { direction: rtl; }

/* خلفية متحركة */
.stApp {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #1c1c1c);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
}

/* أنيميشن الخلفية */
@keyframes gradientBG {
    0% {background-position:0% 50%;}
    50% {background-position:100% 50%;}
    100% {background-position:0% 50%;}
}

/* كروت زجاجية */
.neo {
    background: rgba(255,255,255,0.05);
    border-radius: 20px;
    padding: 20px;
    backdrop-filter: blur(20px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    transition: 0.3s;
}
.neo:hover {
    transform: translateY(-5px);
}

/* عنوان */
.title {
    text-align:center;
    font-size:40px;
    font-weight:bold;
    color:white;
}

/* زر احترافي */
.stButton>button {
    background: linear-gradient(135deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 12px;
    padding: 12px 25px;
    font-size: 18px;
    transition: 0.3s;
}
.stButton>button:hover {
    transform: scale(1.05);
}

/* نص */
h2, h3, h4, p, label {
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🧠 NeuroVision — AI Emotion Dashboard</div>', unsafe_allow_html=True)

# ==========================================================
# 📥 Sidebar (تنظيم احترافي)
# ==========================================================
st.sidebar.header("⚙️ التحكم")

st.sidebar.markdown("### 📥 إدخال البيانات")

image_file = st.sidebar.file_uploader("📷 رفع صورة", type=["jpg","png"])
camera_image = st.sidebar.camera_input("📸 تصوير مباشر")

if camera_image is not None:
    image_file = camera_image

audio_file = st.sidebar.file_uploader("🎤 رفع صوت", type=["mp3","wav","m4a"])

st.sidebar.markdown("### 🎙 تسجيل صوت")

st.sidebar.components.v1.html("""
<script>
let mediaRecorder;
let audioChunks = [];

async function startRecording() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.start();

    mediaRecorder.ondataavailable = event => {
        audioChunks.push(event.data);
    };

    mediaRecorder.onstop = () => {
        const blob = new Blob(audioChunks, { type: 'audio/wav' });
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = 'recorded_audio.wav';
        a.click();
    };
}
function stopRecording() {
    mediaRecorder.stop();
}
</script>

<button onclick="startRecording()">🎙 بدء</button>
<button onclick="stopRecording()">⏹ إيقاف</button>
""", height=120)

st.sidebar.info("بعد التسجيل سيتم تحميل الصوت — ارفعه في الأعلى")

run = st.sidebar.button("🚀 بدء التحليل")

# ==========================================================
# 📷 تحليل الصورة
# ==========================================================
def analyze_image(img):
    API_URL = "https://router.huggingface.co/hf-inference/models/trpakov/vit-face-expression"
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
        "Content-Type": "application/octet-stream"
    }
    return requests.post(API_URL, headers=headers, data=img).json()

# ==========================================================
# 🎤 تحليل الصوت
# ==========================================================
def analyze_audio(audio_bytes):
    aai.settings.api_key = ASSEMBLYAI_API_KEY

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        path = tmp.name

    try:
        transcript = aai.Transcriber().transcribe(path)
        text = transcript.text or ""

        y, sr = librosa.load(path, sr=None)

        energy = float(np.mean(librosa.feature.rms(y=y)))
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        valid = pitches[magnitudes > np.median(magnitudes)]
        pitch = np.mean(valid) if len(valid) > 0 else 0

        tone = {
            "energy": energy,
            "tempo": tempo,
            "pitch": pitch
        }

        return text, tone

    finally:
        if os.path.exists(path):
            os.remove(path)

# ==========================================================
# 🧠 تحليل AI
# ==========================================================
def generate_diagnosis(emotion, text, tone):

    prompt = f"""
أنت نظام تحليل نفسي متقدم.

المعطيات:
- تعبير الوجه: {emotion}
- النص: {text}
- الطاقة: {tone['energy']}
- السرعة: {tone['tempo']}
- النبرة: {tone['pitch']}

حلل بدقة شديدة وقدم:
1. الحالة النفسية الأساسية
2. نسبة كل شعور
3. تفسير نفسي عميق
4. احتمالات القلق أو التوتر
5. توصيات دقيقة

اكتب بالعربية.
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0.7
    )

    return res.choices[0].message.content

# ==========================================================
# 📊 الرسم البياني
# ==========================================================
def emotion_chart(emotions):
    labels = [e['label'] for e in emotions]
    values = [round(e['score']*100,2) for e in emotions]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=labels,
        y=values,
        text=[str(v)+"%" for v in values],
        textposition='auto'
    ))

    fig.update_layout(
        title="📊 نسب المشاعر",
        xaxis_title="الشعور",
        yaxis_title="النسبة",
        template="plotly_dark"
    )

    return fig

# ==========================================================
# 🚀 التشغيل
# ==========================================================
if run:

    if image_file and audio_file:

        progress = st.progress(0)

        progress.progress(20)
        img_bytes = image_file.getvalue()

        progress.progress(40)
        aud_bytes = audio_file.getvalue()

        progress.progress(60)
        emotions = analyze_image(img_bytes)

        progress.progress(75)
        text, tone = analyze_audio(aud_bytes)

        dominant = max(emotions, key=lambda x: x['score'])

        progress.progress(90)
        diagnosis = generate_diagnosis(
            dominant['label'], text, tone
        )

        progress.progress(100)

        # ==================================================
        # 🧠 Dashboard
        # ==================================================
        st.markdown("## 📊 لوحة التحكم")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="neo">', unsafe_allow_html=True)
            st.image(Image.open(BytesIO(img_bytes)))
            st.metric("الحالة", dominant['label'])
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="neo">', unsafe_allow_html=True)
            st.plotly_chart(emotion_chart(emotions), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ==================================================
        # 🎤 الصوت
        # ==================================================
        st.markdown("## 🎤 تحليل الصوت")

        c1, c2, c3 = st.columns(3)
        c1.metric("الطاقة", f"{tone['energy']:.4f}")
        c2.metric("السرعة", f"{tone['tempo']:.2f}")
        c3.metric("النبرة", f"{tone['pitch']:.2f}")

        # ==================================================
        # 📋 التفاصيل
        # ==================================================
        st.markdown("## 📊 تفاصيل المشاعر")

        for e in emotions:
            st.progress(e['score'], text=f"{e['label']} {round(e['score']*100,2)}%")

        # ==================================================
        # 🧠 التشخيص
        # ==================================================
        st.markdown("## 🧠 التشخيص الذكي")
        st.markdown(f"<div class='neo'>{diagnosis}</div>", unsafe_allow_html=True)

    else:
        st.warning("⚠ الرجاء إدخال صورة وصوت")