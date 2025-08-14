from flask import Flask, render_template, request, jsonify
import random
import joblib
from predict import predict_fraud  # Import your custom prediction function

app = Flask(__name__)

# Load model (only needed if you want to directly use the model in this file)
model = joblib.load("fraud_detection_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Get user inputs
            trans_type = request.form['type']
            amount = float(request.form['amount'])
            oldbalanceOrg = float(request.form['oldbalanceOrg'])
            newbalanceOrig = float(request.form['newbalanceOrig'])
            oldbalanceDest = float(request.form['oldbalanceDest'])
            newbalanceDest = float(request.form['newbalanceDest'])

            # Automatically assign step and isFlaggedFraud
            step = random.randint(1, 95)  # Random step between 1 and 95
            isFlaggedFraud = 0  # Always set to 0

            # Create transaction dictionary
            transaction_data = {
                "step": step,
                "type": trans_type,
                "amount": amount,
                "oldbalanceOrg": oldbalanceOrg,
                "newbalanceOrig": newbalanceOrig,
                "oldbalanceDest": oldbalanceDest,
                "newbalanceDest": newbalanceDest,
                "isFlaggedFraud": isFlaggedFraud
            }

            # Use your custom predict function
            prediction = predict_fraud(transaction_data)

            # Or alternatively, directly use the loaded model
            # features = [step, trans_type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, isFlaggedFraud]
            # prediction = model.predict([features])[0]

            return render_template("index.html", prediction=prediction)

        except Exception as e:
            return jsonify({"error": str(e)}), 400

    return render_template("index.html")

if __name__ == "__main__":
    app.run()
