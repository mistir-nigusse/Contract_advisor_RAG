from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
@app.route('/get_question', methods=['POST'])
def get_question():
    data = request.json
    question = data.get('question', '')

    if not question.strip():
        return jsonify({"error": "Question cannot be empty"}), 400

    return jsonify({"message": f"Received: {question}"}), 200

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001)
