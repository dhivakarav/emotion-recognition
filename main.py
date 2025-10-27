from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from fer import FER

app = Flask(__name__)
detector = FER()

def get_personalized_message(emotion):
    messages = {
        'angry': "It seems you're a little upset. Take a short break or try some relaxation exercises.",
        'disgust': "If something feels off, don't hesitate to ask for help or take a breather!",
        'fear': "Feeling uneasy? Deep breaths and a quick stretch might help focus better.",
        'happy': "Great to see you happy and engaged! Keep up the good learning pace.",
        'sad': "If you're feeling down, try changing your study topic or reach out for support.",
        'surprise': "That moment of surprise might mean you learned something new! Awesome!",
        'neutral': "Keep steady and focused. You're doing well!",
        None: "Unable to determine emotion. Keep calm and carry on!"
    }
    return messages.get(emotion.lower() if emotion else None, "Keep up the good work!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    img_bytes = file.read()
    np_img = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'Invalid image'}), 400

    emotion, score = detector.top_emotion(img)
    message = get_personalized_message(emotion)

    return jsonify({
        'emotion': emotion,
        'score': score,
        'message': message
    })

if __name__ == '__main__':
    app.run(debug=True)
