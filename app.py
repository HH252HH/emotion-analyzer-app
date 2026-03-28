# ==========================================================
# 🧠 Emotion AI Pro — NeuroVision (Ultra Dashboard Edition)
# 🎯 تحليل شبه طبي + واجهة أسطورية + دعم عربي كامل
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
# 🎨 تصميم احترافي
# ==========================================================
st.set_page_config(page_title="NeuroVision", layout="wide")

st.markdown("""
<style>
body {
    direction: rtl;
}
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
.neo {
    background: rgba(255,255,255,0.05);
    border-radius: 20px;
    padding: 25px;
    backdrop-filter: blur(15px);
    box-shadow: 0 10px 40px rgba(0,0,0,0.4);
}
h1 {text-align:center;}
</style>
""", unsafe_allow_html=True)

st.title("🧠 NeuroVision — تحليل عاطفي متقدم")

# ==========================================================
# 📥 الإدخال
# ==========================================================
col1, col2 = st.columns(2)

with col1:
    image_file = st.file_uploader("📷 صورة", type=["jpg","png"])

with col2:
    audio_file = st.file_uploader("🎤 صوت", type=["mp3","wav","m4a"])

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
# 🎤 تحليل الصوت (احترافي)
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
# 🧠 تحليل AI متقدم (شبه طبي)
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
# 📊 رسم بياني عربي احترافي
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
        xaxis_title="نوع الشعور",
        yaxis_title="النسبة المئوية",
        font=dict(family="Arial"),
    )

    return fig

# ==========================================================
# 🚀 التشغيل
# ==========================================================
if st.button("🚀 تحليل متقدم"):

    if image_file and audio_file:

        with st.spinner("🧠 تحليل عميق جارٍ..."):

            img_bytes = image_file.getvalue()
            aud_bytes = audio_file.getvalue()

            emotions = analyze_image(img_bytes)
            text, tone = analyze_audio(aud_bytes)

            dominant = max(emotions, key=lambda x: x['score'])

            diagnosis = generate_diagnosis(
                dominant['label'], text, tone
            )

        # ==================================================
        # 🧠 لوحة التحكم الأسطورية
        # ==================================================

        st.markdown("## 🧠 لوحة التحليل المتقدمة")

        colA, colB = st.columns([1,1])

        with colA:
            st.image(Image.open(BytesIO(img_bytes)))
            st.metric("الحالة الأساسية", dominant['label'])

        with colB:
            st.plotly_chart(emotion_chart(emotions), use_container_width=True)

        # ==================================================
        # 🎤 تحليل الصوت التفصيلي
        # ==================================================

        st.markdown("## 🎤 تحليل الصوت")
        st.write(f"الطاقة: {tone['energy']:.4f}")
        st.write(f"السرعة: {tone['tempo']:.2f}")
        st.write(f"النبرة: {tone['pitch']:.2f}")

        # ==================================================
        # 📋 نسب المشاعر
        # ==================================================

        st.markdown("## 📊 تفاصيل دقيقة")

        for e in emotions:
            st.write(f"{e['label']} → {round(e['score']*100,2)}%")

        # ==================================================
        # 🧠 التحليل الذكي
        # ==================================================

        st.markdown("## 🧠 التشخيص النفسي")
        st.write(diagnosis)

    else:
        st.warning("⚠ الرجاء إدخال صورة وصوت")
