# Emotion_classification
This objective of this project is to design and implement an end-to-end pipeline for emotion classification using speech data.
The model was trained on the Ravdess dataset for speech and song of different actors.

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


