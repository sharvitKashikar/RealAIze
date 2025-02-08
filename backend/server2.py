from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
import googletrans
from googletrans import Translator
import os
from dotenv import load_dotenv
from flask_cors import CORS
import logging
from functools import lru_cache

# ✅ Load environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_KEY")

# ✅ Configure Gemini API
genai.configure(api_key=API_KEY)

# ✅ Setup Flask app and CORS
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend interaction

# ✅ Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Initialize Translator
translator = Translator()

class NewsAuthenticator:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-pro')

    @lru_cache(maxsize=128)  # Cache results to avoid redundant API calls for the same article
    def verify_news(self, article_text):
        """Verify news authenticity using Gemini and return a confidence score"""
        prompt = f"""
        Analyze this news article and provide a clear assessment of its authenticity.
        Respond with ONLY the authenticity percentage as an integer and a confidence score out of 100.
        Article: {article_text}
        """
        
        try:
            response = self.model.generate_content(prompt)
            logger.info(f"Gemini API Response: {response.text.strip()}")
            authenticity, confidence = self._parse_response(response.text.strip())
            return authenticity, confidence
        except Exception as e:
            logger.error(f"Error in verifying news: {e}")
            return None, None

    def _parse_response(self, response_text):
        """Parse the response text to extract authenticity and confidence"""
        try:
            logger.info(f"Raw response: {response_text}")
            if 'Authenticity' in response_text:
                authenticity = self._extract_value(response_text, 'Authenticity')
                confidence = self._extract_value(response_text, 'Confidence')
                return authenticity, confidence
            else:
                logger.warning(f"Unexpected response format: {response_text}")
                return None, None
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return None, None

    def _extract_value(self, response_text, label):
        """Extract numerical value after a specific label"""
        try:
            start = response_text.find(label)
            if start != -1:
                start = response_text.find(":", start) + 1
                end = response_text.find("%", start)
                if start != -1 and end != -1:
                    return int(response_text[start:end].strip())
            return None
        except Exception as e:
            logger.error(f"Error extracting value for {label}: {e}")
            return None

news_authenticator = NewsAuthenticator()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/fakenews/verify', methods=['POST'])
def verify_news():
    data = request.json
    article_text = data.get("article_text")

    if not article_text:
        return jsonify({"error": "No article text provided"}), 400
    
    # ✅ Detect language
    detected_lang = translator.detect(article_text).lang

    # ✅ Translate to English if necessary
    if detected_lang != 'en':
        translated_text = translator.translate(article_text, dest='en').text
        logger.info(f"Translated from {detected_lang} to English: {translated_text}")
    else:
        translated_text = article_text

    # ✅ Verify news authenticity with the translated text
    authenticity, confidence = news_authenticator.verify_news(translated_text)

    if authenticity is None:
        return jsonify({"error": "Error verifying news authenticity"}), 500
    
    return jsonify({
        "original_text": article_text,
        "detected_language": detected_lang,
        "translated_text": translated_text,
        "authenticity": authenticity,
        "confidence": confidence
    })

if __name__ == '__main__':
    app.run(debug=True, port=5002)