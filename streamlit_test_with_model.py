import streamlit as st


st.set_page_config(page_title="ToneClone", page_icon="🔊", layout="wide")

import os
import librosa
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from pydub import AudioSegment
from pydub.utils import which
from PIL import Image
import torch
from wav_classifier import (
    spectrogramCNN,
    process_wav_for_model,
    NUM_CLASSES,
    MODEL_LOAD_PATH,
    LABEL_NAMES,
    SAMPLE_LENGTH,
    OVERLAP,
    TARGET_SAMPLE_RATE
)

# Check if ffmpeg is installed
ffmpeg_path = which("ffmpeg")
ffprobe_path = which("ffprobe")


custom_css = """
<style>
    /* Full-screen background */
    .stApp {
        background-color: black;
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

    /* Apply Poppins font */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    body {
        font-family: 'Poppins', sans-serif;
    }

    
    /* Navigation Bar */
    .topnav {
        background-color: #2f3542;
        overflow: hidden;
        position: fixed;
        top: 0;
        width: 100%;
        padding: 10px 0;
        z-index: 1000;
        text-align: center;
    }

    /* Navigation links */
    .topnav a {
        display: inline-block;
        color: white;
        text-align: center;
        padding: 10px 20px;
        text-decoration: none;
        font-size: 18px;
        font-weight: bold;
        transition: 0.3s;
    }

    /* Hover effect */
    .topnav a:hover {
        background-color: #575fcf;
        border-radius: 5px;
    }

    /* Page content spacing */
    .page-content {
        padding-top: 60px; /* Prevents content from being hidden behind the navbar */
    }


    /* Sidebar Styling */
    .stSidebar {
        background-color: #2f3544 !important;
    }

    /* Ensure all sidebar text is visible */
    .stSidebar, .stSidebarContent, .stSidebar label, .stSidebar div {
        color: white !important;
    }

    /* Make sidebar radio buttons and labels visible */
    div[data-baseweb="radio"] label {
        color: white !important;
        font-weight: bold;
    }

    /* Ensure radio button text is visible */
    .stRadio label {
        color: white !important;
        font-weight: bold;
    }

    /* Selected radio button styling */
    div[data-baseweb="radio"] input:checked + label {
        color: #FFD700 !important; /* Gold color for selection */
        font-weight: bold;
    }

    /* Hover effect for visibility */
    div[data-baseweb="radio"] label:hover {
        color: #ffcc00 !important;
        cursor: pointer;
    }

    /* Fix for button visibility */
    .stButton button {
        color: white !important;
        background-color: #ff6600 !important; /* Orange button */
        border-radius: 5px;
        font-weight: bold;
    }

    /* Fix for sidebar expander text */
    .st-expander {
        color: white !important;
    }

    /* Fix for selectbox dropdown */
    .stSelectbox label {
        color: white !important;
    }

    <style>
        /* Navigation Bar */
        .topnav {
            background-color: #444;
            overflow: hidden;
            position: fixed;
            top: 40px;
            width: 100%;
            height: 50px;
            padding: 5px 0;
            z-index: 1000;
            text-align: center;
        }

        /* Navigation links */
        .topnav a {
            display: inline-block;
            color: white;
            text-align: center;
            padding: 14px 20px;
            text-decoration: none;
            font-size: 18px;
            font-weight: bold;
            transition: 0.3s;
        }

        /* Hover effect */
        .topnav a:hover {
            background-color: #ff6600;
            border-radius: 5px;
        }

        /* Page content spacing */
        .page-content {
            padding-top: 60px; /* Prevents content from being hidden behind the navbar */
        }


</style>

"""


st.markdown(custom_css, unsafe_allow_html=True)

st.markdown(
    """
    <style>
        /* Remove extra white space at the top */
        body, .stApp {
            margin: 0;
            padding: 0;
            overflow-x: hidden;
        }

        /* Custom Top Navigation Bar */
        .topnav {
            background-color: #444; /* Dark Gray */
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 45px; /* Adjusted height */
            padding: 5px 0;
            z-index: 9999;
            text-align: center;
            box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.2); /* Adds a slight shadow */
        }

        /* Navigation links */
        .topnav a {
            display: inline-block;
            color: white;
            text-align: center;
            padding: 10px 15px;
            text-decoration: none;
            font-size: 16px;
            font-weight: bold;
            transition: 0.3s;
        }

        /* Hover effect */
        .topnav a:hover {
            background-color: #ff6600;
            border-radius: 5px;
        }

        /* Push page content down so it's not covered */
        .page-content {
            padding-top: 60px; /* Push content further down */
        }
    </style>

    <div class="topnav">
        <a href="#home">Home</a>
        <a href="#record">Record Audio</a>
        <a href="#upload">Upload & Crop Audio</a>
        <a href="#about">About</a>
    </div>
    <div class="page-content"></div>
    """,
    unsafe_allow_html=True
)


# Sidebar Navigation
pages = ["Home", "Record Audio", "Upload & Crop Audio"]
selected_page = st.sidebar.radio("Go to", pages)

# Default image path
DEFAULT_IMAGE_PATH = "home.JPG"

if selected_page == "Home":
    st.markdown(
        """
        <style>
            .home-text {
                font-size: 24px; 
                line-height: 1.6; 
                text-align: justify; /* Optional: Justifies text */
                color: white; 
            }
            .title-text {
                font-size: 50px;
                font-weight: bold;
                text-align: center;
                color: white;
            }
        </style>
        <div class="title-text">ToneClone</div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
            <b style="color: #00CED1;"><i>Overcoming the Learning Curve for Guitar Effects</i></b><br>

            <b style="color: #fcfcfc;">Many beginner guitarists struggle with achieving professional-quality sounds using effects, especially when trying to 
            emulate popular musicians. Distortion, delay, reverb, and modulation effect are used to create professional sounds but can 
            be confusing to new players.</b><br><br>

            <b style="color: #00CED1;">How ToneClone Helps:</b><br>
            <b style="color: #fcfcfc;">ToneClone addresses this challenge by analyzing guitar audio to identify the effects used and provides accessible, tailored 
            guidance to educate guitarists about effects. This approach allows users to bridge the gap between hearing and recreating 
            professional-quality sounds, offering a unique combination of analysis and education that is not available in other products. 
            A simple and intuitive method for new guitar players to analyze guitar tones and receive a step by step instruction to replicate.</b><br><br><br>

        """,
        unsafe_allow_html=True
    )

    st.write("""
    <b style="color: #edbb24;">Navigate the sidebar to record or upload an audio.</b><br>
             """,
        unsafe_allow_html=True)

# Directory for Audio Files
AUDIO_DIR = "uploaded_audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

# Record Audio Page
if selected_page == "Record Audio":
    st.title("🎙 Record Your Audio")
    audio_url = st.text_input("Paste an audio file URL:")
    if audio_url:
        st.audio(audio_url)

# Upload & Crop Audio Page
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
        fig.add_trace(go.Scatter(x=time_axis, y=y, mode="lines", name="Waveform"))
        
        # Initial Cropping Positions
        start_time = st.slider("Start Time (seconds)", 0.0, duration, 0.0, step=0.1)
        end_time = st.slider("End Time (seconds)", start_time, duration, duration, step=0.1)
        
        fig.add_vline(x=start_time, line_width=2, line_dash="dash", line_color="red")
        fig.add_vline(x=end_time, line_width=2, line_dash="dash", line_color="green")
        
        fig.update_layout(
            title="Interactive Audio Waveform",
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude",
            template="plotly_dark",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # ─── CLASSIFICATION BUTTONS ────────────────────────────────────────────────
        # Classify the full (original) audio file using your CNN model.
        if st.button("🧠 Classify Full Audio"):
            # Set up the PyTorch device and load the model from disk
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = spectrogramCNN(NUM_CLASSES).to(device)
            model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=device))
            model.eval()
            
            # Process the full audio file into spectrogram segments
            spectrograms = process_wav_for_model(str(file_path))
            if spectrograms.shape[0] == 0:
                st.error("Audio file is too short for processing.")
            else:
                spectrograms = spectrograms.to(device)
                with torch.no_grad():
                    outputs = model(spectrograms)
                    predictions = torch.sigmoid(outputs)
                    binary_preds = (predictions > 0.5).cpu().numpy()
                
                # Loop over each segment and display detected effects
                segments = binary_preds.shape[0]
                results = []
                for i in range(segments):
                    detected_effects = [LABEL_NAMES[j] for j in range(NUM_CLASSES) if binary_preds[i][j] == 1]
                    # Calculate the time range of each segment
                    step_size = SAMPLE_LENGTH * (1 - OVERLAP / 100)
                    segment_start = i * step_size
                    segment_end = segment_start + SAMPLE_LENGTH
                    if detected_effects:
                        result_str = f"Segment {i+1} ({segment_start:.1f}s - {segment_end:.1f}s): " + ", ".join(detected_effects)
                    else:
                        result_str = f"Segment {i+1} ({segment_start:.1f}s - {segment_end:.1f}s): None"
                    results.append(result_str)
                
                st.write("Prediction on Full Audio:")
                for r in results:
                    st.write(r)

        # Classify the cropped audio segment using your CNN model.
        if st.button("🧠 Classify Cropped Audio"):
            # Crop the audio using the slider positions and export as a temporary WAV file
            start_ms, end_ms = int(start_time * 1000), int(end_time * 1000)
            cropped_audio = audio[start_ms:end_ms]
            temp_cropped_path = Path(AUDIO_DIR) / f"temp_cropped_{uploaded_file.name}"
            cropped_audio.export(temp_cropped_path, format="wav")
            
            # Check if the cropped audio meets the minimum sample length
            y_cropped, sr_cropped = librosa.load(temp_cropped_path, sr=None)
            cropped_duration = librosa.get_duration(y=y_cropped, sr=sr_cropped)
            if cropped_duration < SAMPLE_LENGTH:
                st.error("Cropped audio is shorter than the required sample length for classification. Please select a longer segment.")
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = spectrogramCNN(NUM_CLASSES).to(device)
                model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=device))
                model.eval()
                
                spectrograms = process_wav_for_model(str(temp_cropped_path))
                if spectrograms.shape[0] == 0:
                    st.error("Cropped audio is too short for processing.")
                else:
                    spectrograms = spectrograms.to(device)
                    with torch.no_grad():
                        outputs = model(spectrograms)
                        predictions = torch.sigmoid(outputs)
                        binary_preds = (predictions > 0.5).cpu().numpy()
                    
                    segments = binary_preds.shape[0]
                    results = []
                    for i in range(segments):
                        detected_effects = [LABEL_NAMES[j] for j in range(NUM_CLASSES) if binary_preds[i][j] == 1]
                        step_size = SAMPLE_LENGTH * (1 - OVERLAP / 100)
                        segment_start = i * step_size
                        segment_end = segment_start + SAMPLE_LENGTH
                        if detected_effects:
                            result_str = f"Segment {i+1} ({segment_start:.1f}s - {segment_end:.1f}s): " + ", ".join(detected_effects)
                        else:
                            result_str = f"Segment {i+1} ({segment_start:.1f}s - {segment_end:.1f}s): None"
                        results.append(result_str)
                    
                    st.write("Prediction on Cropped Audio:")
                    for r in results:
                        st.write(r)

        # ─── AUDIO PREVIEW & DOWNLOAD ─────────────────────────────────────────────
        # Preview Cropped Audio
        if st.button("🎧 Preview Crop"):
            start_ms, end_ms = int(start_time * 1000), int(end_time * 1000)
            cropped_audio = audio[start_ms:end_ms]
            preview_path = Path(AUDIO_DIR) / f"preview_{uploaded_file.name}"
            cropped_audio.export(preview_path, format="wav")
            st.audio(str(preview_path))
            st.success("🔊 Previewing cropped audio.")
        
        # Save Cropped Audio
        if st.button("✂️ Save & Download"):
            cropped_file_path = Path(AUDIO_DIR) / f"cropped_{uploaded_file.name}"
            cropped_audio = audio[int(start_time * 1000):int(end_time * 1000)]
            cropped_audio.export(cropped_file_path, format="wav")
            st.success(f"✅ Cropped audio saved: {cropped_file_path.name}")
            with open(cropped_file_path, "rb") as f:
                st.download_button("⬇️ Download Cropped Audio", data=f, file_name=f"cropped_{uploaded_file.name}", mime="audio/wav")
