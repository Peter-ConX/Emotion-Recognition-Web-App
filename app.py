import os
import io
import base64
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image
import numpy as np
from face_emotions import EmotionRecognizer


UPLOAD_FOLDER = 'datasets/uploads'
LABELED_FOLDER = 'datasets/labeled'
MODEL_PATH = 'models/emotion_model.h5'


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# instantiate recognizer (loads model)
recognizer = EmotionRecognizer(model_path=MODEL_PATH)


@app.route('/')
def index():
return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
# expects JSON with key 'image' containing base64 PNG/JPEG data
data = request.json
if not data or 'image' not in data:
return jsonify({'error': 'no image provided'}), 400


img_b64 = data['image'].split(',')[-1]
img_bytes = base64.b64decode(img_b64)
img = Image.open(io.BytesIO(img_bytes)).convert('RGB')


# save raw upload for dataset
ts = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
filename = f'{ts}.jpg'
save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
img.save(save_path)


# get prediction
pred = recognizer.predict_image(np.array(img))


# optionally save labeled to datasets/labeled/<emotion>/
return jsonify(pred)


@app.route('/upload_label', methods=['POST'])
def upload_label():
# save a labeled image into datasets/labeled/<emotion>/ for future training
data = request.json
if not data or 'image' not in data or 'label' not in data:
return jsonify({'error': 'image and label required'}), 400


label = data['label']
img_b64 = data['image'].split(',')[-1]
img_bytes = base64.b64decode(img_b64)
img = Image.open(io.BytesIO(img_bytes)).convert('RGB')


target_dir = os.path.join('datasets', 'labeled', label)
os.makedirs(target_dir, exist_ok=True)
ts = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
fname = os.path.join(target_dir, f'{label}_{ts}.jpg')
img.save(fname)


return jsonify({'status': 'saved', 'path': fname})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
