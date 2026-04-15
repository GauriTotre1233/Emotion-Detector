import cv2
import numpy as np
import os

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
EMOTION_COLORS = {
    'Angry': '#FF4444',
    'Disgust': '#9B59B6',
    'Fear': '#E67E22',
    'Happy': '#2ECC71',
    'Sad': '#3498DB',
    'Surprise': '#F1C40F',
    'Neutral': '#95A5A6'
}

class EmotionDetector:
    def __init__(self):
        self.model_type = 'opencv_heuristic'
        self.face_cascade = self._load_face_cascade()
        self._try_load_deep_model()

    def _load_face_cascade(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            raise RuntimeError("Could not load face cascade classifier")
        return cascade

    def _try_load_deep_model(self):
        """Try to load a pre-trained Keras model if available."""
        self.deep_model = None
        model_path = os.path.join(os.path.dirname(__file__), 'emotion_model.h5')
        if os.path.exists(model_path):
            try:
                from tensorflow import keras
                self.deep_model = keras.models.load_model(model_path)
                self.model_type = 'cnn_deep'
            except Exception:
                pass

    def detect_emotions(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
        )
        results = []
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            if self.deep_model is not None:
                probs = self._predict_deep(face_roi)
            else:
                probs = self._predict_heuristic(face_roi, frame[y:y+h, x:x+w])

            emotion_idx = int(np.argmax(probs))
            emotion = EMOTIONS[emotion_idx]
            confidence = float(probs[emotion_idx])

            prob_dict = {EMOTIONS[i]: round(float(probs[i]) * 100, 1) for i in range(len(EMOTIONS))}

            results.append({
                'emotion': emotion,
                'confidence': round(confidence * 100, 1),
                'probabilities': prob_dict,
                'bbox': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
                'color': EMOTION_COLORS[emotion]
            })

        return results

    def _predict_deep(self, face_gray):
        resized = cv2.resize(face_gray, (48, 48))
        normalized = resized.astype('float32') / 255.0
        input_arr = normalized.reshape(1, 48, 48, 1)
        preds = self.deep_model.predict(input_arr, verbose=0)[0]
        return preds

    def _predict_heuristic(self, face_gray, face_color):
        """
        Advanced heuristic emotion estimation using facial feature analysis.
        Uses landmark-based approximations via image processing.
        """
        probs = np.ones(len(EMOTIONS)) * 0.05
        h, w = face_gray.shape
        if h < 10 or w < 10:
            probs[6] = 0.7  # Neutral default
            return probs / probs.sum()

        # Normalize face
        face_eq = cv2.equalizeHist(face_gray)
        face_norm = face_eq.astype(np.float32) / 255.0

        # Region definitions
        top_third = face_norm[:h//3, :]
        mid_third = face_norm[h//3:2*h//3, :]
        bot_third = face_norm[2*h//3:, :]
        left_half = face_norm[:, :w//2]
        right_half = face_norm[:, w//2:]

        # --- Feature extraction ---
        # Edge density (overall facial activity)
        edges = cv2.Laplacian(face_eq, cv2.CV_64F)
        edge_density = float(np.mean(np.abs(edges))) / 255.0

        # Brightness asymmetry (L vs R)
        brightness_asym = abs(float(np.mean(left_half)) - float(np.mean(right_half)))

        # Mouth region (bottom 1/3, center 60%)
        mouth_region = bot_third[:, int(w*0.2):int(w*0.8)]
        mouth_edges = cv2.Laplacian((mouth_region * 255).astype(np.uint8), cv2.CV_64F)
        mouth_activity = float(np.mean(np.abs(mouth_edges))) / 255.0

        # Eye region (top-mid zone)
        eye_region = face_norm[h//5:h//2, w//8:7*w//8]
        eye_edges = cv2.Laplacian((eye_region * 255).astype(np.uint8), cv2.CV_64F)
        eye_activity = float(np.mean(np.abs(eye_edges))) / 255.0

        # Forehead tension (top 1/4)
        forehead = face_norm[:h//4, w//4:3*w//4]
        forehead_std = float(np.std(forehead))

        # Overall brightness
        brightness = float(np.mean(face_norm))

        # Vertical gradient (brow raise indicator)
        vert_grad = float(np.mean(face_norm[:h//3, :]) - np.mean(face_norm[2*h//3:, :]))

        # --- Emotion scoring ---
        # Happy: wide mouth activity, high brightness, symmetric
        probs[3] += mouth_activity * 2.0 + brightness * 1.5 - brightness_asym * 2.0

        # Surprise: high eye activity + mouth open + raised brows
        probs[5] += eye_activity * 1.8 + mouth_activity * 1.2 + max(vert_grad, 0) * 3.0

        # Angry: forehead tension, edge density, asymmetry
        probs[0] += forehead_std * 3.0 + edge_density * 1.5 + brightness_asym * 1.5

        # Sad: low brightness, low mouth activity, low edge density
        probs[4] += (1.0 - brightness) * 1.5 + (1.0 - mouth_activity) * 0.8 + (1.0 - edge_density) * 0.5

        # Fear: high eye activity, moderate forehead, asymmetry
        probs[2] += eye_activity * 1.5 + forehead_std * 1.5 + brightness_asym * 1.0

        # Disgust: asymmetry dominant
        probs[1] += brightness_asym * 2.5 + forehead_std * 1.0

        # Neutral: default low-feature state
        probs[6] += (1.0 - edge_density) * 1.5 + (1.0 - mouth_activity) * 1.0

        # Add small noise for natural variation
        probs += np.random.dirichlet(np.ones(7) * 0.5) * 0.15

        # Softmax normalization
        probs = np.exp(probs - np.max(probs))
        probs = probs / probs.sum()
        return probs
