from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

# Define route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    person_age = int(request.form['person_age'])
    person_income = int(request.form['person_income'])
    loan_amnt = int(request.form['loan_amnt'])
    credit_score = int(request.form['credit_score'])
    person_gender = request.form['person_gender']
    person_education = request.form['person_education']
    loan_intent = request.form['loan_intent']
    person_home_ownership = request.form['person_home_ownership']

    # Create a DataFrame for the input
    input_data = pd.DataFrame([{
        'person_age': person_age,
        'person_income': person_income,
        'loan_amnt': loan_amnt,
        'credit_score': credit_score,
        'person_gender': person_gender,
        'person_education': person_education,
        'loan_intent': loan_intent,
        'person_home_ownership': person_home_ownership
    }])

    # Make prediction
    prediction = model.predict(input_data)
    result = "Loan Approved" if prediction[0] == 1 else "Loan Not Approved"

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
