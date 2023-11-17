import os
from flask import Flask, render_template, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
image_model = load_model('path/to/your/image/model.h5')  # Load your trained image deepfake detection model
video_model = load_model('path/to/your/video/model.h5')  # Load your trained video deepfake detection model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        if file_path.endswith('.mp4'):
            result = detect_video_deepfake(file_path)
        else:
            result = detect_image_deepfake(file_path)

        return render_template('index.html', result=result)

def detect_image_deepfake(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)

    prediction = image_model.predict(img)
    if prediction > 0.5:  # You might need to adjust this threshold based on your model output
        return "Real Image"
    else:
        return "Deepfake Image"

def detect_video_deepfake(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    result = ""
    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (256, 256))
        frame = frame / 255.0  # Normalize pixel values
        frame = np.expand_dims(frame, axis=0)

        prediction = video_model.predict(frame)
        if prediction > 0.5:  # You might need to adjust this threshold based on your model output
            result = "Real Video"
        else:
            result = "Deepfake Video"

    cap.release()
    return result

if __name__ == '__main__':
    app.run(debug=True)
