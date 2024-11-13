from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import base64
from io import BytesIO
from PIL import Image
import numpy as np

app = Flask(__name__)
CORS(app)

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

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    images_base64 = data.get("img", [])

    if not images_base64:
        return jsonify({"error": "No images provided"}), 400

    results = []

    for img_str in images_base64:
        try:
            img_data = img_str.split(",")[1]


            img_bytes = base64.b64decode(img_data)

            img = Image.open(BytesIO(img_bytes)).convert('RGB')


            img_np = np.array(img)


            demography = DeepFace.analyze(img_np)

            serializable_result = make_serializable(demography)
            results.append(serializable_result)
        except Exception as e:
            results.append({"error": str(e)})

    return jsonify(results)

