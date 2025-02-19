import wave
import numpy as np
import librosa
import torch
import torch.nn as nn
import datetime

SAMPLE_LENGTH = 10  # In seconds
OVERLAP = 0  # Percentage overlap
N_FFT = 2048
HOP_LENGTH = 512
NUM_MELS = 128
NUM_CLASSES = 13
TARGET_SAMPLE_RATE = 32000
MODEL_LOAD_PATH = "/Users/jennt/Desktop/210/final_multi_effects.mod"


EFFECT_LABELS = {
    "CLN": "clean", "ODV": "overdrive", "DST": "distortion", "FUZ": "fuzz", "TRM": "tremolo",
    "PHZ": "phaser", "FLG": "flanger", "CHR": "chorus", "DLY": "delay", "HLL": "hall_reverb",
    "PLT": "plate_reverb", "OCT": "octaver", "FLT": "auto_filter"
}
LABEL_NAMES = list(EFFECT_LABELS.values())  # Ordered list of effect names
NUM_CLASSES = len(LABEL_NAMES)

class spectrogramCNN(nn.Module):
    def __init__(self, num_classes):
        super(spectrogramCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten_size = (512 * (128 // 32) * (626 // 32))

        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        x = self.pool(torch.relu(self.bn5(self.conv5(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x) # removed sigmoid, redundant if using BCEWithLogitcsLoss
        return x
    

def split_wav(file_path, sample_length=SAMPLE_LENGTH, overlap=OVERLAP):
    with wave.open(file_path, 'rb') as wav:
        num_channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        print(f"sample rate is {sample_rate}")
        num_frames = wav.getnframes()

        audio_data = np.frombuffer(wav.readframes(num_frames), dtype=np.int16)
        if num_channels == 2:
            audio_data = audio_data.reshape(-1, 2).mean(axis=1)  # Convert stereo to mono

        # Resample only if needed
        if sample_rate != TARGET_SAMPLE_RATE:
            print(f"Resampling from {sample_rate} Hz to {TARGET_SAMPLE_RATE} Hz...")
            audio_data = librosa.resample(audio_data.astype(np.float32), orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)
            sample_rate = TARGET_SAMPLE_RATE  # Update sample rate

        samples_per_segment = sample_rate * sample_length
        overlap_samples = int(sample_rate * (overlap / 100))
        step_size = samples_per_segment - overlap_samples

        segments = []
        start = 0
        while start + samples_per_segment <= len(audio_data):
            segment_data = audio_data[start:start + samples_per_segment]
            segments.append(segment_data)
            start += step_size

        return segments, sample_rate

def generate_spectrogram(audio_segment, sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH, num_mels=NUM_MELS):
    # Normalize audio to range [-1, 1]
    audio_float = audio_segment.astype(np.float32) / (np.max(np.abs(audio_segment)) + np.finfo(np.float32).eps)

    sgrm = librosa.feature.melspectrogram(y=audio_float, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=num_mels)
    return librosa.amplitude_to_db(sgrm, ref=np.max)

def process_wav_for_model(wav_file):
    print(f"Processing file: {wav_file}")

    segments, sample_rate = split_wav(wav_file)

    spectrograms = []
    for segment in segments:
        sgrm_db = generate_spectrogram(segment, sample_rate)
        spectrograms.append(sgrm_db)

    spectrograms = np.array(spectrograms)  # Convert list
    spectrograms = torch.tensor(spectrograms, dtype=torch.float32).unsqueeze(1)  # Add channel dim for CNN

    return spectrograms


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load model
# model = spectrogramCNN(NUM_CLASSES).to(device)
# model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=device))
# model.eval()

# # Process WAV file
# wav_file = '/content/drive/MyDrive/Capstone 210/For Classification/CherubRock_real_unisolated.wav'
# spectrograms = process_wav_for_model(wav_file).to(device)

# # Get step size for timestamp calculations
# step_size = int(TARGET_SAMPLE_RATE * SAMPLE_LENGTH * (1 - OVERLAP / 100))

# print("Spectrogram shape before model:", spectrograms.shape)

# # Make predictions
# with torch.no_grad():
#     outputs = model(spectrograms)  # Raw logits before sigmoid activation
#     predictions = torch.sigmoid(outputs)  # Sigmoid activation

# # Convert to binary predictions
# binary_preds = (predictions > 0.5).cpu().numpy()
# logits = outputs.cpu().numpy()

# # format seconds to MM:SS
# def format_time(seconds):
#     return str(datetime.timedelta(seconds=int(seconds)))[2:]  # Removes hours if <1hr

# # Print raw logits and labeled predictions
# for i, (segment_logits, segment_pred) in enumerate(zip(logits, binary_preds)):
#     sigmoid_probs = torch.sigmoid(torch.tensor(segment_logits)).numpy()

#     # Calculate start and end time of the segment
#     start_time = (i * step_size) / TARGET_SAMPLE_RATE
#     end_time = start_time + (SAMPLE_LENGTH)

#     timestamp_range = f"{format_time(start_time)} - {format_time(end_time)}"

#     print(f"\nSegment {i+1} ({timestamp_range}):")
#     #print("Raw Logits:", segment_logits)
#     #print("Sigmoid Probabilities:", sigmoid_probs)

#     detected_effects = [LABEL_NAMES[j] for j in range(NUM_CLASSES) if segment_pred[j] == 1]
#     if detected_effects:
#         print(f"Predicted Effects: {', '.join(detected_effects)}")
#     else:
#         print("Predicted Effects: None")