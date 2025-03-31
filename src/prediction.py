import os
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define constants
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "language_identification_model.h5")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.json")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")

def load_model_and_metadata():
    """Load the trained model, tokenizer, and metadata."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"Tokenizer file not found: {TOKENIZER_PATH}")

    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")

    # Load model
    model = load_model(MODEL_PATH)

    # Load tokenizer
    with open(TOKENIZER_PATH, "r") as f:
        tokenizer_json = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json)

    # Load metadata
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    return model, tokenizer, metadata

def preprocess_text(text, tokenizer, max_length):
    """Preprocess text for prediction."""
    text = text.lower().strip()  # Lowercase and strip spaces
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length)
    return padded_sequence

def predict_language(text):
    """Predict the language of the given text."""
    # Load model and metadata
    model, tokenizer, metadata = load_model_and_metadata()

    # Preprocess input text
    processed_text = preprocess_text(text, tokenizer, metadata["max_length"])

    # Predict
    prediction = model.predict(processed_text)[0]
    predicted_index = prediction.argmax()
    predicted_lang = metadata["language_codes"][predicted_index]
    confidence = prediction[predicted_index]

    # Get top 3 language predictions
    top_indices = prediction.argsort()[-3:][::-1]
    top_languages = [(metadata["language_codes"][i], float(prediction[i])) for i in top_indices]

    # Print results
    print(f"Predicted Language: {predicted_lang} (Confidence: {confidence:.4f})")
    print("Top 3 Predictions:")
    for lang, conf in top_languages:
        print(f"  - {lang}: {conf:.4f}")

    return {
        "predicted_language": predicted_lang,
        "confidence": float(confidence),
        "top_languages": top_languages
    }

# Example Usage (Uncomment below to test in a script)
text = "omo wetin dey happen"
predict_language(text)
