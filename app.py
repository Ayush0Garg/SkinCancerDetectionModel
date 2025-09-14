import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

# --- Step 1: FastAPI app + CORS ---
app = FastAPI(title="Skin Cancer Detection API")

# Use your frontend URL in production
origins = ["https://glittery-pothos-3fee6b.netlify.app"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Step 2: Skin Cancer Model ---
num_classes = 7
base_model = MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
base_model.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=outputs)

# Load weights
weights_path = "model_weights.weights.h5"
try:
    model.load_weights(weights_path)
    print("Skin cancer model weights loaded successfully!")
except Exception as e:
    print(f"Error loading skin cancer model weights: {e}")

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

# --- Step 4: Pretrained ImageNet Model for Skin Check ---
skin_check_model = MobileNetV2(weights='imagenet')
skin_related_keywords = ['hand', 'arm', 'leg', 'face', 'torso', 'person', 'human']

# --- Step 5: Predict endpoint ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224, 224))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process image: {e}")

    # --- Step 5a: Check if image contains human skin/body ---
    img_array_check = np.expand_dims(np.array(img), axis=0)
    img_array_check = preprocess_input(img_array_check)
    preds_check = skin_check_model.predict(img_array_check)
    top_preds = decode_predictions(preds_check, top=3)[0]
    is_skin = any(any(keyword in pred[1].lower() for keyword in skin_related_keywords) for pred in top_preds)
    
    if not is_skin:
        return {"class": "Not human skin detected", "confidence": 0, "risk": "N/A"}

    # --- Step 5b: Skin cancer prediction ---
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    preds = model.predict(img_array)
    pred_class = int(np.argmax(preds, axis=1)[0])
    confidence = float(preds[0][pred_class])

    return {
        "class": class_indices[pred_class],
        "confidence": round(confidence * 100, 2),
        "risk": risk_mapping[class_indices[pred_class]]
    }
