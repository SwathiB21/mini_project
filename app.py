import os
from flask import Flask, render_template, request, redirect
import numpy as np
import tensorflow as tf
import cv2

# Set TensorFlow to ignore GPU devices
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices([], 'GPU')

app = Flask(__name__)

# Load your model
model = tf.keras.models.load_model('model.h5', compile=False)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_grayscale(image):
    if len(image.shape) < 3:
        return True
    b, g, r = cv2.split(image)
    if np.array_equal(b, g) and np.array_equal(g, r):
        return True
    else:
        return False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if not is_grayscale(img_cv):
            return render_template('result.html', prediction_result='Error: This is Inappropriate Image. Please Upload a Chest X-ray.')

        img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.reshape(img, (1, 224, 224, 3))
        
        prediction_result = predict_image(img)
        return render_template('result.html', prediction_result=prediction_result)
    
    return redirect(request.url)

def predict_image(image):
    prediction = model.predict(image)
    if prediction[0][0] < 0.5:
        prediction_result = 'COVID Affected'
    else:
        prediction_result = 'COVID Unaffected'
    return prediction_result

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
