from flask import Flask, request, jsonify, render_template, send_from_directory, Response
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import base64
import time
import os
import gdown
from flask_cors import CORS

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'resnet50v2_model.keras')

# Flask app config
app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)
CORS(app)
# Google Drive model download
MODEL_FILE_ID = "1UInMiIbaHChmI-KSQ7VRMp_53RZpSDd4"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load the model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    model_loaded = True
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model_loaded = False

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define mouth and eye landmarks (MediaPipe's 468 landmarks)
MOUTH_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Default eye threshold
EYE_THRESHOLD = 0.25

# Timers for drowsiness detection
EYES_CLOSED_START_TIME = None
YAWNING_START_TIME = None
EYES_CLOSED_DURATION = 1 # seconds
YAWNING_DURATION = 3  # seconds

# Initialize webcam (change index to 1 for external webcam)

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print(f"Failed to open camera {1}, trying another index...")
    cap = cv2.VideoCapture(0)  # Fallback to default camera if external fails



def extract_mouth(frame, landmarks, w, h):
    points = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in MOUTH_LANDMARKS])
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    margin = 10  # Add some margin88
    x_min, y_min = max(0, x_min - margin), max(0, y_min - margin)
    x_max, y_max = min(w, x_max + margin), min(h, y_max + margin)
    mouth_img = frame[y_min:y_max, x_min:x_max]
    if mouth_img.size == 0:
        return None
    mouth_img = cv2.resize(mouth_img, (224, 224)) / 255.0
    return np.expand_dims(mouth_img, axis=0)




def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return float((A + B) / (2.0 * C))  # Convert to Python float


def convert_to_native_types(obj):
    if isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return convert_to_native_types(obj.tolist())
    else:
        return obj


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        while True:
            success, frame = cap.read()
            if not success:
                break
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/api/status')
def api_status():
    return jsonify({
        'status': 'API is running',
        'model_loaded': bool(model_loaded),
        'endpoints': {
            'calibrate': '/api/calibrate',
            'detect': '/api/detect'
        }
    })


@app.route('/api/calibrate', methods=['POST'])
def calibrate():
    global EYE_THRESHOLD
    ear_values = []

    if 'frames' not in request.json:
        return jsonify({'error': 'No frames provided'}), 400

    frames = request.json['frames']
    for frame_data in frames:
        try:
            encoded_data = frame_data.split(',')[1]
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    left_eye = np.array(
                        [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE])
                    right_eye = np.array(
                        [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE])
                    ear_values.append(float(
                        (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0))
        except Exception as e:
            print(f"Error processing frame: {str(e)}")

    if len(ear_values) > 0:
        EYE_THRESHOLD = float(np.mean(ear_values) - 0.01)
        return jsonify({'threshold': float(EYE_THRESHOLD)})
    else:
        return jsonify({'error': 'No face detected in calibration frames'}), 400


@app.route('/api/detect', methods=['POST'])
def detect_drowsiness():
    global EYES_CLOSED_START_TIME, YAWNING_START_TIME

    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400

    try:
        encoded_data = request.json['image'].split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        response = {'drowsy': False, 'yawning': False, 'eyes_closed': False, 'face_detected': False}

        if results.multi_face_landmarks:
            response['face_detected'] = True
            for face_landmarks in results.multi_face_landmarks:
                mouth_img = extract_mouth(frame, face_landmarks.landmark, w, h)
                is_yawning = False

                if mouth_img is not None:
                    yawning_prob = model.predict(mouth_img, verbose=0)[0][0]
                    is_yawning = bool(yawning_prob > 0.5)

                    if is_yawning:
                        if YAWNING_START_TIME is None:
                            YAWNING_START_TIME = time.time()
                        elif time.time() - YAWNING_START_TIME >= YAWNING_DURATION:
                            response['drowsy'] = True
                    else:
                        YAWNING_START_TIME = None

                    response['yawning'] = is_yawning
                    response['yawn_probability'] = float(yawning_prob)

                left_eye = np.array(
                    [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE])
                right_eye = np.array(
                    [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE])
                avg_ear = float((eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0)
                response['eye_aspect_ratio'] = float(avg_ear)
                response['eyes_closed'] = avg_ear < EYE_THRESHOLD

                if response['eyes_closed']:
                    if EYES_CLOSED_START_TIME is None:
                        EYES_CLOSED_START_TIME = time.time()
                    elif time.time() - EYES_CLOSED_START_TIME >= EYES_CLOSED_DURATION:
                        response['drowsy'] = True
                else:
                    EYES_CLOSED_START_TIME = None

        return jsonify(convert_to_native_types(response))

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
