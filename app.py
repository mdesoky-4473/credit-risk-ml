from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Placeholder logic: always return the same score
    return jsonify({
        "risk_score": 0.82,
        "decision": "approve"
    })

@app.route('/', methods=['GET'])
def home():
    return "Credit Risk ML API is live!"

if __name__ == '__main__':
    app.run(debug=True)
