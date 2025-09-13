from flask import Flask, request, jsonify, render_template, send_from_directory, Response
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import base64
import time
import os
import gdown
import logging
from datetime import datetime
from flask_cors import CORS
from config import get_config

# Load configuration
config = get_config()

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'resnet50v2_model.keras')

# Flask app config
app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)
app.config.from_object(config)

# Configure CORS
CORS(app, origins=config.CORS_ORIGINS)

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Google Drive model download
MODEL_FILE_ID = config.MODEL_FILE_ID
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Check if model exists, if not provide instructions
if not os.path.exists(MODEL_PATH):
    logger.warning("Model file not found!")
    logger.info(f"Please download the model manually from: {MODEL_URL}")
    logger.info("Save it as: resnet50v2_model.keras in the 'model' folder")
    logger.info("Continuing without model - calibration and basic detection will still work")

# Load the model
model = None
model_loaded = False

if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        model_loaded = True
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        model_loaded = False
else:
    logger.warning("Model file not found - yawning detection will be disabled")
    logger.info("Eye closure detection will still work using MediaPipe")

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define mouth and eye landmarks (MediaPipe's 468 landmarks)
MOUTH_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Default eye threshold
EYE_THRESHOLD = config.DEFAULT_EYE_THRESHOLD

# Timers for drowsiness detection
EYES_CLOSED_START_TIME = None
YAWNING_START_TIME = None
EYES_CLOSED_DURATION = config.EYES_CLOSED_DURATION  # seconds
YAWNING_DURATION = config.YAWNING_DURATION  # seconds

# Initialize webcam with better error handling
def initialize_camera():
    """Initialize camera with fallback options"""
    camera_indices = config.CAMERA_INDICES
    
    for idx in camera_indices:
        try:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                # Test if we can read a frame
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✅ Camera {idx} initialized successfully")
                    return cap
                else:
                    cap.release()
            else:
                cap.release()
        except Exception as e:
            print(f"❌ Error with camera {idx}: {str(e)}")
            continue
    
    print("❌ No working camera found!")
    print("💡 Please check:")
    print("   - Camera is connected and working")
    print("   - No other applications are using the camera")
    print("   - Camera permissions are granted")
    return None

cap = initialize_camera()



def extract_mouth(frame, landmarks, w, h):
    points = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in MOUTH_LANDMARKS])
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    margin = 10  # Add some margin
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
        if cap is None:
            # Return a placeholder image if no camera
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "No Camera Available", (150, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', placeholder)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            return
            
        consecutive_failures = 0
        max_failures = config.MAX_CAMERA_FAILURES
        
        while True:
            try:
                success, frame = cap.read()
                if not success:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        # Camera seems to be disconnected, return error frame
                        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(error_frame, "Camera Disconnected", (150, 240), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        ret, buffer = cv2.imencode('.jpg', error_frame)
                        frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                        break
                    continue
                
                consecutive_failures = 0  # Reset failure counter on success
                
                # Validate frame
                if frame is None or frame.size == 0:
                    continue
                    
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                    
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                       
            except Exception as e:
                print(f"Error in video feed: {str(e)}")
                # Return error frame
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, "Camera Error", (200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                ret, buffer = cv2.imencode('.jpg', error_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                break

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/api/status')
def api_status():
    return jsonify({
        'status': 'API is running',
        'model_loaded': bool(model_loaded),
        'camera_available': cap is not None,
        'endpoints': {
            'calibrate': '/api/calibrate',
            'detect': '/api/detect',
            'health': '/api/health'
        }
    })

@app.route('/api/health')
def health_check():
    """Comprehensive health check endpoint"""
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'components': {
            'flask_app': 'healthy',
            'model': 'healthy' if model_loaded else 'degraded',
            'camera': 'healthy' if cap is not None else 'unavailable',
            'mediapipe': 'healthy'
        },
        'configuration': {
            'eye_threshold': EYE_THRESHOLD,
            'eyes_closed_duration': EYES_CLOSED_DURATION,
            'yawning_duration': YAWNING_DURATION,
            'max_frames_per_calibration': config.MAX_FRAMES_PER_CALIBRATION
        }
    }
    
    # Check if any critical components are down
    if not model_loaded and cap is None:
        health_status['status'] = 'degraded'
    elif not model_loaded or cap is None:
        health_status['status'] = 'degraded'
    
    return jsonify(health_status)


@app.route('/api/calibrate', methods=['POST'])
def calibrate():
    global EYE_THRESHOLD
    ear_values = []

    # Input validation
    if not request.json:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    if 'frames' not in request.json:
        return jsonify({'error': 'No frames provided'}), 400

    frames = request.json['frames']
    
    # Validate frames data
    if not isinstance(frames, list) or len(frames) == 0:
        return jsonify({'error': 'Invalid frames data - must be non-empty array'}), 400
    
    if len(frames) > config.MAX_FRAMES_PER_CALIBRATION:  # Prevent excessive data
        return jsonify({'error': f'Too many frames provided - maximum {config.MAX_FRAMES_PER_CALIBRATION} allowed'}), 400
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

    # Input validation
    if not request.json:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    if 'image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400
    
    # Check if image data is valid
    try:
        image_data = request.json['image']
        if not image_data or not isinstance(image_data, str):
            return jsonify({'error': 'Invalid image data'}), 400
    except Exception as e:
        return jsonify({'error': f'Error parsing image data: {str(e)}'}), 400

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

                # Yawning detection logic
                if mouth_img is not None and model is not None and model_loaded:
                    try:
                        yawning_prob = model.predict(mouth_img, verbose=0)[0][0]
                        is_yawning = bool(yawning_prob > 0.5)
                    except Exception as e:
                        print(f"Model prediction error: {str(e)}")
                        # Fallback to heuristic method
                        mouth_points = np.array([(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in MOUTH_LANDMARKS])
                        mouth_height = np.max(mouth_points[:, 1]) - np.min(mouth_points[:, 1])
                        mouth_width = np.max(mouth_points[:, 0]) - np.min(mouth_points[:, 0])
                        mouth_ratio = mouth_height / mouth_width if mouth_width > 0 else 0
                        is_yawning = mouth_ratio > 0.3
                        yawning_prob = mouth_ratio
                else:
                    # If no model available, use a simple heuristic based on mouth opening
                    mouth_points = np.array([(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in MOUTH_LANDMARKS])
                    mouth_height = np.max(mouth_points[:, 1]) - np.min(mouth_points[:, 1])
                    mouth_width = np.max(mouth_points[:, 0]) - np.min(mouth_points[:, 0])
                    mouth_ratio = mouth_height / mouth_width if mouth_width > 0 else 0
                    is_yawning = mouth_ratio > 0.3  # Simple heuristic
                    yawning_prob = mouth_ratio

                # Handle yawning timer logic
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
    # Try different ports to avoid conflicts
    ports_to_try = [config.PORT, config.PORT + 1, config.PORT + 2, 8000, 8080]
    
    logger.info(f"Starting DriveSafe Drowsiness Detection System")
    logger.info(f"Configuration: {config.__class__.__name__}")
    logger.info(f"Debug mode: {config.DEBUG}")
    
    for port in ports_to_try:
        try:
            logger.info(f"Starting Flask server on port {port}...")
            app.run(host=config.HOST, port=port, debug=config.DEBUG)
            break
        except OSError as e:
            if "Address already in use" in str(e):
                logger.warning(f"Port {port} is already in use, trying next port...")
                continue
            else:
                logger.error(f"Error starting server on port {port}: {str(e)}")
                break
    else:
        logger.error("Could not find an available port. Please check your system.")
