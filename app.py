from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import base64
from io import BytesIO
import numpy as np
import tensorflow as tf
from keras.models import load_model

app = Flask(__name__)
CORS(app, origins=["https://ceol.vercel.app"])


def emotion_analysis(emotions):
    objects = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    y_pos = np.arange(len(objects))

objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
model = load_model('model.h5')

def make_serializable(obj):
    """Recursively convert non-serializable elements (e.g., numpy data types) to Python native types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(i) for i in obj]
    return obj

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "https://ceol.vercel.app"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

@app.route('/analyze', methods=['OPTIONS','POST'])
def analyze():
    if request.method == 'OPTIONS':
        response = make_response('', 204)
        response.headers["Access-Control-Allow-Origin"] = "https://ceol.vercel.app"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return response
        
    data = request.json
    images_base64 = data.get("img", [])

    if not images_base64:
        return jsonify({"error": "No images provided"}), 400

    results = []

    for img_str in images_base64:
        try:
            img_data = img_str.split(",")[1]

            img_bytes = base64.b64decode(img_data)

            img_temp = BytesIO(img_bytes)

            img = tf.keras.utils.load_img(img_temp, color_mode="grayscale", target_size=(48, 48))
            x = tf.keras.utils.img_to_array(img)
            x = np.expand_dims(x, axis=0)

            x /= 255

            custom = model.predict(x)
            emotion_analysis(custom[0])

            x = np.array(x, 'float32')
            x = x.reshape([48, 48]);

            m = 0.000000000000000000001
            a = custom[0]
            for i in range(0, len(a)):
                if a[i] > m:
                    m = a[i]
                    ind = i

            serializable_result = make_serializable(objects[ind])
            results.append(serializable_result)
        except Exception as e:
            results.append({"error": str(e)})

    return jsonify(results)
