import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
import numpy as np
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

# --- Step 1: Model Architecture ---
num_classes = 7
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=outputs)

# --- Step 2: Load Weights ---
weights_path = "model_weights.weights.h5"
model.load_weights(weights_path)

# --- Step 3: Class Labels + Risk ---
class_indices = {
    0: 'Melanocytic nevi (NV)',
    1: 'Melanoma (MEL)',
    2: 'Benign keratosis (BKL)',
    3: 'Basal cell carcinoma (BCC)',
    4: 'Actinic keratoses (AKIEC)',
    5: 'Vascular lesion (VASC)',
    6: 'Dermatofibroma (DF)'
}

risk_mapping = {
    'Melanocytic nevi (NV)': 'Benign, not cancerous',
    'Melanoma (MEL)': 'Malignant, dangerous',
    'Benign keratosis (BKL)': 'Benign, low risk',
    'Basal cell carcinoma (BCC)': 'Malignant, but low metastatic risk',
    'Actinic keratoses (AKIEC)': 'Precancerous, moderate risk',
    'Vascular lesion (VASC)': 'Benign, low risk',
    'Dermatofibroma (DF)': 'Benign, not cancerous'
}

# --- Step 4: FastAPI App ---
app = FastAPI(title="Skin Cancer Detection API")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224, 224))

    # Preprocess
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    pred_class = np.argmax(preds, axis=1)[0]
    confidence = float(preds[0][pred_class])

    return {
        "class": class_indices[pred_class],
        "confidence": round(confidence * 100, 2),
        "risk": risk_mapping[class_indices[pred_class]]
    }
