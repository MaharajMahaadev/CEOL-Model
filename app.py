import gradio as gr
import base64
from io import BytesIO
import numpy as np
import tensorflow as tf
from keras.models import load_model

# Load the pre-trained model
model = load_model('model.h5')

# Define the emotion categories
objects = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def emotion_analysis(emotions):
    objects = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    y_pos = np.arange(len(objects))

# Function to handle image prediction and emotion analysis
def analyze(img_base64):
    try:
        # Decode the base64 image
        img_data = img_base64.split(",")[1]
        img_bytes = base64.b64decode(img_data)
        img_temp = BytesIO(img_bytes)

        # Preprocess the image
        img = tf.keras.utils.load_img(img_temp, color_mode="grayscale", target_size=(48, 48))
        x = tf.keras.utils.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x /= 255  # Normalize the image data

        # Predict emotions
        predictions = model.predict(x)
        emotion_analysis(predictions[0])

        # Find the most probable emotion
        max_probability = max(predictions[0])
        emotion_index = np.argmax(predictions[0])
        predicted_emotion = objects[emotion_index]

        return {
            "emotion": predicted_emotion,
            "probability": float(max_probability)
        }
    except Exception as e:
        return {"error": str(e)}

# Define the Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("### Emotion Detection")
    with gr.Row():
        image_input = gr.Textbox(label="Input Base64 Image String")
        output = gr.JSON(label="Output Emotion")

    analyze_button = gr.Button("Analyze")
    analyze_button.click(fn=analyze, inputs=image_input, outputs=output)

# Launch the app
demo.launch()
