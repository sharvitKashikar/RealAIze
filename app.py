from flask import Flask, request, jsonify, render_template
import requests

app = Flask(__name__)

DEEPFAKE_API_URL = "http://127.0.0.1:5001/deepfake/detect"
FAKE_NEWS_API_URL = "http://127.0.0.1:5002/fakenews/verify"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/deepfake/detect", methods=["POST"])
def detect_deepfake():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    
    file = request.files['file']
    files = {'file': (file.filename, file.stream, file.mimetype)}

    response = requests.post(DEEPFAKE_API_URL, files=files)
    return jsonify(response.json())

@app.route("/fakenews/verify", methods=["POST"])
def verify_news():
    data = request.json
    if "article_text" not in data:
        return jsonify({"error": "No article text provided"}), 400

    response = requests.post(FAKE_NEWS_API_URL, json=data)
    return jsonify(response.json())

if __name__ == "__main__":
    app.run(debug=True, port=5000)
