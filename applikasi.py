from flask import Flask, render_template, Response, session, flash, Blueprint, request, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
import bcrypt
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os
import requests
import pymysql

app = Flask(__name__)
app.secret_key = '6LdhAeYpAAAAAKfhQ9GP6zirlMQZuZQCs-W93Z-T'

# Konfigurasi MySQL
conn = pymysql.connect(host='localhost',
                       user='root',
                       password='',
                       database='flask_app_klasifikasklng',
                       cursorclass=pymysql.cursors.DictCursor)

# Blueprint untuk klasifikasi
klasifikasi_bp = Blueprint('klasifikasi', __name__)

# Fungsi untuk memeriksa apakah jenis file yang diunggah diizinkan
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Fungsi untuk melakukan prediksi gambar
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

# Route untuk prediksi
@klasifikasi_bp.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'email' not in session:
        flash("Anda harus login untuk mengakses halaman ini.")
        return redirect(url_for('login_route'))
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

app.register_blueprint(klasifikasi_bp)

@app.route('/')
def home():
    if 'email' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login_route'))

@app.route('/register', methods=['POST', 'GET'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    else:
        name = request.form['name']
        email = request.form['email']
        password = request.form['password'].encode('utf-8')
        hash_password = bcrypt.hashpw(password, bcrypt.gensalt()).decode('utf-8')

        recaptcha_response = request.form['g-recaptcha-response']
        recaptcha_secret = '6LdhAeYpAAAAAKfhQ9GP6zirlMQZuZQCs-W93Z-T'
        data = {'secret': recaptcha_secret, 'response': recaptcha_response}
        response = requests.post('https://www.google.com/recaptcha/api/siteverify', data=data)
        result = response.json()

        if result['success']:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (name, email, password) VALUES (%s, %s, %s)", (name, email, hash_password))
            conn.commit()
            cursor.close()

            session['name'] = name
            session['email'] = email
            return redirect(url_for('login_route'))
        else:
            flash("Verifikasi reCAPTCHA gagal. Silakan coba lagi.")
            return redirect(url_for('register_route'))

@app.route('/login', methods=['GET', 'POST'])
def login_route():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password'].encode('utf-8')

        recaptcha_response = request.form['g-recaptcha-response']
        recaptcha_secret = '6LdhAeYpAAAAAKfhQ9GP6zirlMQZuZQCs-W93Z-T'
        data = {'secret': recaptcha_secret, 'response': recaptcha_response}
        response = requests.post('https://www.google.com/recaptcha/api/siteverify', data=data)
        result = response.json()

        if result['success']:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
            user = cursor.fetchone()
            cursor.close()

            if user and bcrypt.checkpw(password, user['password'].encode('utf-8')):
                session['name'] = user['name']
                session['email'] = user['email']
                return redirect(url_for('home'))
            else:
                flash("Gagal, Email dan Password Tidak Cocok")
                return redirect(url_for('login_route'))
        else:
            flash("Verifikasi reCAPTCHA gagal. Silakan coba lagi.")
            return redirect(url_for('login_route'))
    else:
        return render_template('login.html')

@app.route('/home')
def dashboard():
    if 'email' in session:
        return render_template('home.html', name=session['name'])
    else:
        return redirect(url_for('login_route'))

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
            
            # Display the result on the frame
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            

if __name__ == '__main__':
    app.run(debug=True)
