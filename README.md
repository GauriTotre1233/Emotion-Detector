# AFFECT — Face Emotion Detection

A real-time face emotion detection application using OpenCV, Flask, and a live webcam feed. Detects multiple faces simultaneously, classifies emotions, displays probability spectrums, and persists results to SQLite.

---

## Architecture

```
emotion_detector/
├── backend/
│   ├── app.py                  # Flask API server
│   ├── models/
│   │   └── emotion_model.py    # EmotionDetector (OpenCV + optional CNN)
│   ├── database/
│   │   └── db.py               # SQLite helpers
│   └── utils/
├── frontend/
│   ├── index.html              # Single-page app
│   └── static/
│       ├── css/style.css
│       └── js/app.js
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone / unzip the project

```bash
cd emotion_detector
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the backend server

```bash
cd backend
python app.py
```

The server starts on **http://localhost:5000**

### 5. Open the app

Open your browser and go to: **http://localhost:5000**

Click **▶ START** to activate your webcam and begin real-time emotion detection.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/predict` | Submit a base64 JPEG frame, returns face bounding boxes + emotions |
| `GET` | `/api/history` | Last N detections from SQLite (`?limit=50`) |
| `GET` | `/api/stats` | Aggregated emotion statistics |
| `GET` | `/api/health` | Server + model status |

### Example: POST /api/predict

**Request:**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQ..."
}
```

**Response:**
```json
{
  "faces": [
    {
      "emotion": "Happy",
      "confidence": 78.4,
      "probabilities": {
        "Angry": 2.1, "Disgust": 0.8, "Fear": 1.2,
        "Happy": 78.4, "Sad": 3.5, "Surprise": 9.0, "Neutral": 5.0
      },
      "bbox": { "x": 120, "y": 80, "w": 160, "h": 160 },
      "color": "#00e676"
    }
  ],
  "timestamp": "2025-01-15T14:32:00.123456"
}
```

---

## Emotion Classes

| Emotion | Color |
|---------|-------|
| 😠 Angry | Red |
| 🤢 Disgust | Purple |
| 😨 Fear | Orange |
| 😄 Happy | Green |
| 😢 Sad | Blue |
| 😲 Surprise | Yellow |
| 😐 Neutral | Grey |

---

## Optional: Use a Deep CNN Model

To use a real trained Keras model instead of the built-in heuristic:

1. Download a pre-trained FER-2013 model (`.h5` format, input shape `(48,48,1)`, 7 output classes)
2. Place it at `backend/models/emotion_model.h5`
3. Install TensorFlow: `pip install tensorflow`
4. Restart the server — it will auto-detect and load the model

The server reports `"model": "cnn_deep"` in `/api/health` when a deep model is loaded, otherwise `"opencv_heuristic"`.

---

## Database

Results are saved to `backend/emotion_data.db` (SQLite).

Schema:
```sql
CREATE TABLE detections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT,
    emotion     TEXT,
    confidence  REAL,
    probabilities TEXT,   -- JSON string
    face_count  INTEGER
);
```

---

## License

MIT
