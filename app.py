from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)  

# Load model at the top
model = pickle.load(open("model.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))
print(model_columns)  

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    loan_purpose = data.get("loan_purpose")
    employment_status = data.get("employment_status")

    # One-hot encode loan_purpose (excluding 'debt_consolidation' to avoid multicollinearity)
    loan_purpose_options = [
        "home_improvement","major_purchase","medical_expense","other", 
        "small_business", "vacation"]
    loan_purpose_encoded = [1 if loan_purpose == opt else 0 for opt in loan_purpose_options]

    # One-hot encode employment_status (excluding 'employed')
    employment_status_options = ["self-employed", "unemployed"]
    employment_status_encoded = [1 if employment_status == opt else 0 for opt in employment_status_options]


    # Final feature array
    features = np.array([[
        float(data['income']),
        float(data['age']),
        float(data['credit_score']),
        float(data['loan_amount']),
        float(data['debt_to_income_ratio']),
        *employment_status_encoded,  # [self-employed, unemployed]
        *loan_purpose_encoded        # [other, home_improvement, ..., vacation]
    ]])

    risk_score = model.predict_proba(features)[0][1]
    decision = "approve" if risk_score < 0.5 else "deny"

    return jsonify({
        "decision": decision,
        "risk_score": round(float(risk_score), 2)
    })

if __name__ == '__main__':
    app.run(debug=True)

