import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import sys

model = load_model("final_model.h5")

emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
labelencoder = LabelEncoder()
labelencoder.fit(emotions)

def extract_feature(data, sr):

    stft = np.abs(librosa.stft(data))
    result = np.array([])
    mfccs=librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)                          #mfcss
    mfccs1 = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T, axis=0)
    result = np.hstack((result, mfccs1))
    delta1 = librosa.feature.delta(mfccs)                                         #delta mfcss
    delta2 = librosa.feature.delta(mfccs, order=2)                                #delta square mfcss
    delta1_mean = np.mean(delta1.T, axis=0)
    delta2_mean = np.mean(delta2.T, axis=0)
    result = np.hstack((result,delta1_mean, delta2_mean))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)         #chroma
    result = np.hstack((result, chroma))
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sr).T,axis=0)         #mel
    result = np.hstack((result, mel))
    spec_centr=np.mean(librosa.feature.spectral_centroid(y=data, sr=sr).T, axis=0)
    spec_bw=np.mean(librosa.feature.spectral_bandwidth(y=data, sr=sr).T, axis=0)
    #rolloff = np.mean(librosa.feature.spectral_rolloff(y=data, sr=sr).T, axis=0)
    #flatness = np.mean(librosa.feature.spectral_flatness(y=data).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)          #zcr
    result = np.hstack((result, spec_centr, spec_bw, zcr))
    #result = np.hstack((result,rolloff, flatness))
    rmse = np.mean(librosa.feature.rms(y=data).T, axis=0)                         #rmse
    #print("rmse shape:",rmse.shape)
    result = np.hstack((result, rmse))    
    #tempo, _ = librosa.beat.beat_track(y=data, sr=sr)
    #print("tempo shape:",tempo.shape)
    #result = np.hstack((result, tempo))
    #y_harm, y_perc = librosa.effects.hpss(data)
    #harmonic_energy = np.mean(librosa.feature.rms(y=y_harm).T, axis=0)
    #percussive_energy = np.mean(librosa.feature.rms(y=y_perc).T, axis=0)
    #print("harm shape:",harmonic_energy.shape)
    #print("harm shape:",percussive_energy.shape)
    #result = np.hstack((result, harmonic_energy, percussive_energy))
    #tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(data), sr=sr).T, axis=0)
    #print("tonnetz shape:",tonnetz.shape)
    #result = np.hstack((result, tonnetz))
    #contrast = np.mean(librosa.feature.spectral_contrast(y=data, sr=sr).T, axis=0)
    #print("contrast shape:",contrast.shape)
    #result = np.hstack((result, contrast))
    return result

def predict_emotion(file_path):
    data, sr = librosa.load(file_path, duration=3, offset=0.5)
    features = extract_feature(data, sr)
    features = np.expand_dims(features, axis=0)
    features = np.expand_dims(features, axis=2)
    prediction = model.predict(features)
    predicted_label = labelencoder.inverse_transform([np.argmax(prediction)])[0]
    print(f"[{file_path}] → Predicted Emotion: {predicted_label.upper()}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_model.py <path_to_audio.wav>")
    else:
        for file in sys.argv[1:]:
            predict_emotion(file)
