import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


class EmotionRecognizer:
def __init__(self, model_path='models/emotion_model.h5', face_cascade_path=None, target_size=(48,48)):
self.model_path = model_path
self.target_size = target_size
self.face_cascade_path = face_cascade_path or cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
self.face_cascade = cv2.CascadeClassifier(self.face_cascade_path)
if os.path.exists(model_path):
self.model = load_model(model_path)
self.emotions = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
else:
self.model = None
self.emotions = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']


def _prepare_face(self, gray, x, y, w, h):
face = gray[y:y+h, x:x+w]
face = cv2.resize(face, self.target_size)
face = face.astype('float') / 255.0
face = img_to_array(face)
face = np.expand_dims(face, 0)
return face


def predict_image(self, rgb_image: np.ndarray):
# rgb_image: HxWx3 numpy array
gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))


if len(faces) == 0:
return {'error': 'no_face_detected'}


# take the largest face
faces = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)
x,y,w,h = faces[0]


if self.model is None:
return {'error': 'model_not_loaded'}


face_input = self._prepare_face(gray, x, y, w, h)
preds = self.model.predict(face_input)[0]
top_idx = int(np.argmax(preds))
result = {
'emotion': self.emotions[top_idx],
'confidence': float(preds[top_idx])
}
return result