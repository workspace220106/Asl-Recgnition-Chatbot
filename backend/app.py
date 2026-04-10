from flask import Flask, request, jsonify
from flask_cors import CORS

from preprocess import preprocess_image
from predictor import predict
from chatbot import get_bot_response
from translation import translate_text, translate_batch, get_supported_languages
from autocorrect_service import autocorrect_text

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


# =========================
# TEST ROUTE (optional but useful)
# =========================
@app.route("/")
def home():
    return "Backend running ✅"


# =========================
# SUPPORTED LANGUAGES
# =========================
@app.route("/api/languages", methods=["GET"])
def languages():
    """Return the list of supported languages for the dropdown."""
    langs = get_supported_languages()
    return jsonify(langs)


# =========================
# TRANSLATE ROUTE
# =========================
@app.route("/api/translate", methods=["POST", "OPTIONS"])
def translate_api():
    """
    General-purpose translation endpoint.
    Expects JSON: { "text": "...", "lang": "es" }
    Returns JSON: { "translated_text": "...", "src_lang": "en", "dest_lang": "es" }
    """
    data = request.json or {}
    text = data.get("text", "")
    lang = data.get("lang", "en")

    result = translate_text(text, dest_lang=lang)
    return jsonify(result)


# =========================
# BATCH TRANSLATE ROUTE
# =========================
@app.route("/api/translate_batch", methods=["POST", "OPTIONS"])
def translate_batch_api():
    """
    Batch translation endpoint.
    Expects JSON: { "texts": ["...", "..."], "lang": "es" }
    Returns JSON: { "translated_texts": ["...", "..."] }
    """
    data = request.json or {}
    texts = data.get("texts", [])
    lang = data.get("lang", "en")

    translated_texts = translate_batch(texts, dest_lang=lang)
    return jsonify({"translated_texts": translated_texts})


# =========================
# AUTOCORRECT ROUTE
# =========================
@app.route("/api/autocorrect", methods=["POST", "OPTIONS"])
def autocorrect_api():
    """
    Autocorrect endpoint.
    Expects JSON: { "text": "...", "lang": "en" }
    Returns JSON: { "corrected_text": "..." }
    """
    data = request.json or {}
    text = data.get("text", "")
    lang = data.get("lang", "en")

    corrected_text = autocorrect_text(text, lang=lang)
    return jsonify({"corrected_text": corrected_text})


# =========================
# PREDICT ROUTE
# =========================
@app.route("/predict", methods=["POST"])
def predict_api():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    lang = request.form.get("lang", "en")  # Optional language from frontend

    img = preprocess_image(file.read())
    if img is None:
        reply_text = "Sorry, I couldn't detect a hand clearly."
        if lang != "en":
            reply_text = translate_text(reply_text, dest_lang=lang)["translated_text"]
        return jsonify({
            "prediction": "Unknown",
            "confidence": 0.0,
            "chat_reply": reply_text,
            "lang": lang
        })

    label, confidence = predict(img)

    THRESHOLD = 0.50  # Balanced threshold for high-accuracy model

    if confidence < THRESHOLD:
        label = "Unknown"
        reply = "I'm not exactly sure about that sign. Could you try again closer to the light?"
    else:
        reply = get_bot_response(label)

    # Translate the chat reply if a non-English language is selected
    translated_reply = reply
    if lang != "en":
        translated_reply = translate_text(reply, dest_lang=lang)["translated_text"]

    return jsonify({
        "prediction": str(label),
        "confidence": round(confidence, 3),
        "chat_reply": translated_reply,
        "chat_reply_original": reply,
        "lang": lang
    })


# =========================
# DICTIONARY ROUTE
# =========================
@app.route("/api/dictionary", methods=["GET"])
def get_dictionary():
    import os
    # Read available letters from the static dictionary folder
    dict_dir = os.path.join(app.root_path, "static", "dictionary")
    if not os.path.exists(dict_dir):
        return jsonify([])

    # Get all .jpg file base names
    letters = [f.split(".")[0] for f in os.listdir(dict_dir) if f.endswith(".jpg")]
    letters.sort()
    return jsonify(letters)


# =========================
# CHATBOT ROUTE
# =========================
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get("message", "")
    lang = data.get("lang", "en")

    reply = get_bot_response(message)

    # Translate the chatbot reply if needed
    translated_reply = reply
    if lang != "en":
        translated_reply = translate_text(reply, dest_lang=lang)["translated_text"]

    return jsonify({
        "reply": translated_reply,
        "reply_original": reply,
        "lang": lang
    })


# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(debug=True)
