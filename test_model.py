import os
import sys
import numpy as np
import pandas as pd
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import zipfile
import tempfile
import shutil

model = load_model("final_model.h5")  
emotion_labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
encoder = LabelEncoder()
encoder.fit(emotion_labels)
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

def prediction(audio_path):
    try:
        y, sr = librosa.load(audio_path, duration=3, offset=0.5)  
        feats = extract_feature(y,sr)
        if feats is None:
            print(f"[INFO] Skipped {audio_path} due to feature issue.")
            return
        x_input = np.expand_dims(feats, axis=0)
        x_input = np.expand_dims(x_input, axis=2)  
        preds = model.predict(x_input)
        predicted_class = np.argmax(preds)
        emotion = encoder.inverse_transform([predicted_class])[0]
        print(f"> {audio_path} => Emotion: {emotion.upper()}")
    except Exception as err:
        print(f"[ERROR] Problem with file {audio_path}: {err}")
def proccess_files(args_list):
    for item in args_list:
        if item.endswith('.zip') and os.path.isfile(item):                            #zip folder containing multiple .wav files
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    with zipfile.ZipFile(item, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    for root, _, files in os.walk(temp_dir):
                        for f in files:
                            if f.endswith(".wav"):
                                file_path = os.path.join(root, f)
                                prediction(file_path)
            except Exception as e:
                print(f"[ERROR] Failed to extract or process zip '{item}': {e}")
        elif item.endswith('.csv'):                                                 #.csv file containing the file paths of .wav files
            try:
                data = pd.read_csv(item)
                col_name = data.columns[0]
                for idx, path in enumerate(data[col_name]):
                    if isinstance(path, str) and path.endswith('.wav') and os.path.isfile(path):
                        prediction(path)
                    else:
                        print(f"  - Skipping line {idx + 1}: {path}")
            except Exception as e:
                print(f"[ERROR] Failed to load CSV '{item}': {e}")
        elif item.endswith('.wav') and os.path.isfile(item):                        #individual .wav files
            prediction(item)
        else:
            print(f"[ERROR] Unsupported file or path not found: {item}")


