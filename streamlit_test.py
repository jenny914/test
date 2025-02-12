import streamlit as st
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pydub import AudioSegment
from PIL import Image
import plotly.graph_objects as go
from pydub.utils import which


AudioSegment.converter = which("ffmpeg")

st.set_page_config(page_title="ToneClone", page_icon="🔊", layout="wide")


# Custom CSS
dark_theme = """
<style>
    /* Full-screen background image */
    .stApp {
        background-color: grey;
        background-size: cover;
    }
    /* Semi-transparent overlay */
    .stApp::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.5); /* 50% Transparent Black */
        z-index: -1;
    }
    /* White text for visibility */
    .stMarkdown, .stTitle, .stHeader, .stText, .stSidebar {
        color: white !important;
    }
</style>
"""

# Apply dark mode styling
st.markdown(dark_theme, unsafe_allow_html=True)

# Sidebar Navigation
pages = ["Home", "Record Audio", "Upload & Crop Audio"]
selected_page = st.sidebar.radio("Go to", pages)

# Default image path
DEFAULT_IMAGE_PATH = "home.JPG"

# Home Page
if selected_page == "Home":
    st.title("🎵 ToneClone")
    st.write("Navigate through the sidebar to record or upload audio.")



# Directory for Audio Files
AUDIO_DIR = "uploaded_audio"
os.makedirs(AUDIO_DIR, exist_ok=True)



    # Check if Image Exists
if os.path.exists(DEFAULT_IMAGE_PATH):
        img = Image.open(DEFAULT_IMAGE_PATH)
        st.image(img, caption="Welcome to ToneClone", use_column_width=True)
else:
        st.warning("⚠️ Image not found! Please place 'home.JPG' in the project folder.")

# 🎙 Record Audio Page
if selected_page == "Record Audio":
    st.title("🎙 Record Your Audio")

    # Use Streamlit's built-in audio player
    audio_url = st.text_input("Paste an audio file URL:")
    if audio_url:
        st.audio(audio_url)
elif selected_page == "Upload & Crop Audio":
    st.title("📤 Upload & Crop Your Audio File")

    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])

    if uploaded_file:
        file_path = Path(AUDIO_DIR) / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"✅ Uploaded: {file_path.name}")
        st.audio(str(file_path))

        # Convert to WAV for Processing
        audio = AudioSegment.from_file(file_path).set_channels(1)
        audio.export(file_path, format="wav")

        # Load audio with Librosa
        y, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        time_axis = np.linspace(0, duration, num=len(y))

        # Create Plotly Waveform
        fig = go.Figure()

        # Add waveform trace
        fig.add_trace(go.Scatter(x=time_axis, y=y, mode="lines", name="Waveform"))

        # Initial Cropping Positions
        start_time = st.slider("Start Time (seconds)", 0.0, duration, 0.0, step=0.1)
        end_time = st.slider("End Time (seconds)", start_time, duration, duration, step=0.1)

        # Add Vertical Markers for Start and End Time
        fig.add_vline(x=start_time, line_width=2, line_dash="dash", line_color="red", name="Start")
        fig.add_vline(x=end_time, line_width=2, line_dash="dash", line_color="green", name="End")

        fig.update_layout(
            title="Interactive Audio Waveform",
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude",
            xaxis=dict(range=[0, duration]),
            template="plotly_dark",
            height=300
        )

        # Display Plot
        st.plotly_chart(fig, use_container_width=True)

        # 🎧 Preview Cropped Audio
        if st.button("🎧 Preview Crop"):
            start_ms, end_ms = int(start_time * 1000), int(end_time * 1000)
            cropped_audio = audio[start_ms:end_ms]

            preview_path = Path(AUDIO_DIR) / f"preview_{uploaded_file.name}"
            cropped_audio.export(preview_path, format="wav")

            st.audio(str(preview_path))
            st.success("🔊 Previewing cropped audio. Adjust if needed.")

        

        # ✂️ Save Cropped Audio
        if st.button("✂️ Save & Download"):
            start_ms, end_ms = int(start_time * 1000), int(end_time * 1000)
            cropped_audio = audio[start_ms:end_ms]

            cropped_file_path = Path(AUDIO_DIR) / f"cropped_{uploaded_file.name}"
            cropped_audio.export(cropped_file_path, format="wav")

            st.success(f"✅ Cropped audio saved: {cropped_file_path.name}")
            st.audio(str(cropped_file_path))

            with open(cropped_file_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Cropped Audio",
                    data=f,
                    file_name=f"cropped_{uploaded_file.name}",
                    mime="audio/wav"
                )
