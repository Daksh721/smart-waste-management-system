import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
CORS(app) 

MODEL_PATH = r"C:\Users\ddaak\OneDrive\Desktop\code\pythons\SmartWaste_Model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = {
    0: "Organic Waste",
    1: "Recyclable Waste"
}

def prepare_image(image_bytes):
    """Resizes and formats the image for the MobileNetV2 model."""
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((224, 224))
    
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        image_bytes = file.read()
        processed_image = prepare_image(image_bytes)
        
        prediction = model.predict(processed_image)
        
        score = float(prediction[0][0])
        
        class_index = 1 if score > 0.5 else 0
        predicted_class = CLASS_NAMES[class_index]
        confidence = score if class_index == 1 else (1.0 - score)

        return jsonify({
            'prediction': predicted_class,
            'confidence': f"{confidence * 100:.2f}%",
            'raw_score': score
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)