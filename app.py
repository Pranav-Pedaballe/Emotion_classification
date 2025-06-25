import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load model
model = load_model("final_model_saved.keras")

# Prepare label encoder (same order as training)
emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
labelencoder = LabelEncoder()
labelencoder.fit(emotions)

# Feature extraction function (same as training)
def extract_feature(data, sr):
    result = np.array([])
    stft = np.abs(librosa.stft(data))
    mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T, axis=0)
    delta1 = librosa.feature.delta(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40))
    delta2 = librosa.feature.delta(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40), order=2)
    result = np.hstack([mfccs, np.mean(delta1.T, axis=0), np.mean(delta2.T, axis=0)])
    result = np.hstack((result, np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)))
    result = np.hstack((result, np.mean(librosa.feature.melspectrogram(y=data, sr=sr).T, axis=0)))
    result = np.hstack((result, np.mean(librosa.feature.spectral_centroid(y=data, sr=sr).T, axis=0)))
    result = np.hstack((result, np.mean(librosa.feature.spectral_bandwidth(y=data, sr=sr).T, axis=0)))
    result = np.hstack((result, np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)))
    result = np.hstack((result, np.mean(librosa.feature.rms(y=data).T, axis=0)))
    return result

# Streamlit UI
st.title("🎙️ Speech Emotion Recognition")
st.write("Upload a `.wav` audio file to predict the emotion.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    with st.spinner('Processing...'):
        try:
            data, sr = librosa.load(uploaded_file, duration=3, offset=0.5)
            features = extract_feature(data, sr)
            features = np.expand_dims(features, axis=0)
            features = np.expand_dims(features, axis=2)
            prediction = model.predict(features)
            predicted_emotion = labelencoder.inverse_transform([np.argmax(prediction)])[0]
            st.success(f"🎉 Predicted Emotion: **{predicted_emotion.upper()}**")
        except Exception as e:
            st.error(f"⚠️ Error processing file: {e}")
