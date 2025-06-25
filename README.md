# Emotion_classification
Link for the app:- https://emotionclassification-9tkwdaauqykmyagrvxvdsn.streamlit.app/

This objective of this project is to design and implement an end-to-end pipeline for emotion classification using speech data.
The model was trained on the RAVDESS dataset which includes speech and song recordings of different actors.

There are 8 emotions which include :- **Angry,Calm, Disgust,Fearful,Happy,Neutral,Sad and Surprised** 

First we load the data and split the data into training and test in the ratio of 80/20

Different audio features were extracted and the features finally used were:-
 - MFCCs(Mel-Frequency Cepstral Coefficients)
 - Delta and Delta-Delta of MFCCs
 - Chroma
 - Mel Spectrogram
 - Spectral Centroid
 - Spectral Bandwidth
 - Zero Crossing Rate(zcr)
 - Root Mean Square Energy (rmse)

We apply data augmentation to the training data by adding noise and shifting the signal right.
Label encoding is then done to the data

##Model 
A CNN model was built with 4 convolutional layers and finally a dense layer was used.

```python
model = Sequential()
model.add(Input(shape=(264, 1)))
model.add(Conv1D(256, 5, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(128, 5, padding='same', kernel_regularizer=l1_l2(...)))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=8))
model.add(Conv1D(128, 5, padding='same', kernel_regularizer=l1_l2(...)))
model.add(Activation('relu'))
model.add(Conv1D(128, 5, padding='same', kernel_regularizer=l1_l2(...)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(8, activation='softmax'))
optim=Adam(learning_rate=0.001)
```
The model was then trained for a maximum of 100 epochs with early stopping.

##Results
![Screenshot 2025-06-25 185709](https://github.com/user-attachments/assets/f62ec8eb-5703-4892-ac98-c3a1e56158c3)

Overall Accuracy:- 87%
F1 score:- 
 - angry 0.88
 - calm 0.94
 - disgust 0.84
 - fearful 0.84
 - happy 0.89
 - neutral 0.82
 - sad 0.84
 - surprised 0.84

Confusion Matrix:-
![Screenshot 2025-06-25 185735](https://github.com/user-attachments/assets/1969f6e0-229a-48bf-a037-14155f72be33)

Accuracy of each class:-

![Screenshot 2025-06-25 190351](https://github.com/user-attachments/assets/b4b2aa14-ae2e-4519-852d-f05fa8b57ffd)


TO test the model:-
 
Process to run locally:-

git clone https://github.com/Pranav-Pedaballe/Emotion_classification.git

cd Emotion_classification

python -m venv env

source env/bin/activate or .\env\Scripts\activate 

pip install -r requirements.txt

to run on the app:-

streamlit run app.py

to test files

python test_model.py






 



