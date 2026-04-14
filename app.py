import os
import cv2
import numpy as np
import base64
import json
import time
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from database.db import init_db, save_detection, get_recent_detections, get_emotion_stats
from models.emotion_model import EmotionDetector

app = Flask(__name__, static_folder='../frontend/static', template_folder='../frontend/templates')
CORS(app)

# Initialize DB and model
init_db()
detector = EmotionDetector()

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), '..', 'frontend')

@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/static/<path:path>')
def static_files(path):
    return send_from_directory(os.path.join(FRONTEND_DIR, 'static'), path)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        # Decode base64 image
        img_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Invalid image'}), 400

        # Detect faces and emotions
        results = detector.detect_emotions(frame)

        # Save to DB
        for r in results:
            save_detection(
                emotion=r['emotion'],
                confidence=r['confidence'],
                probabilities=json.dumps(r['probabilities']),
                face_count=len(results)
            )

        return jsonify({'faces': results, 'timestamp': datetime.now().isoformat()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/history', methods=['GET'])
def history():
    limit = request.args.get('limit', 50, type=int)
    rows = get_recent_detections(limit)
    return jsonify({'history': rows})


@app.route('/api/stats', methods=['GET'])
def stats():
    data = get_emotion_stats()
    return jsonify({'stats': data})


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': detector.model_type, 'timestamp': datetime.now().isoformat()})


if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
