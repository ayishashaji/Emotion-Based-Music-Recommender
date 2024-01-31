# Emotion Based Music Recommender

import streamlit as st
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import webbrowser
import numpy as np
import cv2

# Load model
classifier = load_model(r'C:\DL projects\music_emotion\model.h5')
label = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


try:
    emotion = np.load("emotion.npy")
except:
    emotion = ""

# Load face cascade
try:
    face_cascade = cv2.CascadeClassifier(
        r'C:\DL projects\music_emotion\haarcascade_frontalface_default.xml')
except Exception:
    st.error("Error loading cascade classifiers")

st.header("Emotion Based Music Recommender")

#Video Transformation

class Faceemotion(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)

        output = ""  

        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48),
                                  interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = label[maxindex]
                output = str(finalout)

            label_position = (x, y)
            cv2.putText(img, output, label_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            np.save("emotion.npy", np.array(output))
        print("Detected Emotion:", output)
        return img

# user input

lang = st.text_input('LANGUAGE')
singer = st.text_input("SINGER")

if lang and singer:
    webrtc_streamer(key='key', desired_playing_state=True,
                    video_processor_factory=Faceemotion)

btn = st.button("Recommend me songs")

if btn:
    if not emotion:
        st.warning("Please let me capture your emotion first")
    else:
        # https://www.youtube.com/results?search_query=chithra+sad+song+malayalam
        webbrowser.open(f"https://www.youtube.com/results?search_query={singer}+{emotion}+song+{lang}")
