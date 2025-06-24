import librosa
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import sys
import os

# Load the trained model
model = load_model("final_model.h5")

# Define emotion labels (must match training order)
emotion_labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
labelencoder = LabelEncoder()
labelencoder.fit(emotion_labels)

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

def predict_emotion(file_path):
    try:
        data, sr = librosa.load(file_path, duration=3, offset=0.5)
        features = extract_feature(data, sr)
        features = np.expand_dims(features, axis=0)
        features = np.expand_dims(features, axis=2)
        prediction = model.predict(features)
        predicted_label = labelencoder.inverse_transform([np.argmax(prediction)])
        return predicted_label[0]
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inference.py path_to_audio.wav")
    else:
        file_path = sys.argv[1]
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
        else:
            result = predict_emotion(file_path)
            if result:
                print("Predicted Emotion:", result)
