from flask import Flask, render_template, Response, session, flash, request, redirect, url_for
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os
import logging
from keras.utils import load_img, img_to_array
from keras.models import load_model


app = Flask(__name__)
app.secret_key = '6LdhAeYpAAAAAKfhQ9GP6zirlMQZuZQCs-W93Z-T'

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Function to check if the uploaded file is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Function to predict the image
def predict_image(filepath):
    model = load_model('model.h5')
    img = load_img(filepath, target_size=(300, 300))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    if prediction[0][0] <= 0.5:
        return 'Cacat'
    else:
        return 'Normal'

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'email' not in session:
        flash("Anda harus login untuk mengakses halaman ini.")
        return redirect(url_for('login'))
    else:
        if request.method == 'POST':
            if 'file' not in request.files:
                return render_template('predict.html', message='No file part')
            file = request.files['file']
            if file.filename == '':
                return render_template('predict.html', message='No selected file')
            if file and allowed_file(file.filename):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                result = predict_image(filepath)
                return render_template('predict.html', message=result, image_url=filepath)
            return render_template('predict.html', message='File not allowed')
        else:
            return render_template('predict.html')

@app.route('/')
def home():
    if 'users' not in session:
        session['users'] = {
            "admin@example.com": {"username": "admin", "password": "password123"}
        }

    if 'email' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/register', methods=['POST', 'GET'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    else:
        username = request.form['name']
        email = request.form['email']
        password = request.form['password']

        users = session.get('users', {})
        if email in users:
            flash("Email sudah terdaftar.")
            return redirect(url_for('register'))
        else:
            users[email] = {"username": username, "password": password}
            session['users'] = users
            flash("Registrasi berhasil. Silakan login.")
            return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        users = session.get('users', {})
        user = users.get(email)

        if user and user['password'] == password:
            session['name'] = user['username']
            session['email'] = email
            return redirect(url_for('home'))
        else:
            flash("Gagal, Email dan Password Tidak Cocok")
            return redirect(url_for('login'))
    else:
        return render_template('login.html')

@app.route('/home')
def dashboard():
    if 'email' in session:
        return render_template('home.html', name=session['name'])
    else:
        return redirect(url_for('login'))

model = load_model('model.h5')
class_names = ["Non-Defective", "Defective"]

# Function to preprocess image for model
def preprocess_image(img):
    img = cv2.resize(img, (300, 300))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

# Function to predict class
def predict_class(img):
    img = preprocess_image(img)
    prediction = model.predict(img)
    return class_names[int(prediction[0][0] <= 0.5)]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            prediction = predict_class(frame_rgb)
            label = f"Prediction: {prediction}"
            
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
    app.run(debug=True)
