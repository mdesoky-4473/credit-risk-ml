import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Sample data with string categories
data = pd.DataFrame({
    'income': [50000, 80000, 30000, 90000, 75000, 62000, 55000, 70000],
    'age': [25, 45, 35, 50, 40, 30, 28, 38],
    'credit_score': [600, 720, 580, 680, 640, 690, 710, 670],
    'loan_amount': [10000, 20000, 5000, 30000, 15000, 18000, 12000, 25000],
    'debt_to_income_ratio': [30, 40, 60, 25, 35, 45, 50, 33],
    'employment_status': [
        'employed', 'employed', 'unemployed', 'self-employed',
        'self-employed', 'unemployed', 'employed', 'unemployed'
    ],
    'loan_purpose': [
        'debt_consolidation', 'vacation', 'medical_expense', 'home_improvement',
        'major_purchase', 'small_business', 'other', 'debt_consolidation'
    ],
    'label': [0, 1, 1, 0, 0, 1, 0, 1]
})

# One-hot encode categorical variables (drop_first avoids dummy variable trap)
data = pd.get_dummies(data, columns=['employment_status', 'loan_purpose'], drop_first=True)

# Train model
X = data.drop('label', axis=1)
y = data['label']

model = RandomForestClassifier()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")
