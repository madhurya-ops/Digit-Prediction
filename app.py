from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

# Load the trained model and scaler
model_data = joblib.load("svm_model.pkl")
weights = model_data["weights"]
bias = model_data["bias"]
scaler = model_data["scaler"]


app = Flask(__name__, static_folder="static")

def predict_digit(features):
    features = np.array(features).reshape(1, -1)  # Ensure shape is (1, 784)
    features = scaler.transform(features)  # Scale features

    # Compute SVM decision function
    decision_value = np.dot(features, weights) + bias  # Linear SVM equation
    prediction = np.sign(decision_value)  # Convert to class label

    return int(prediction)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")  # Serve the HTML page

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if "features" not in data:
        return jsonify({"error": "Missing 'features' key"}), 400
    if len(data["features"]) != 784:
        return jsonify({"error": f"Expected 784 features, got {len(data['features'])}"}), 400

    prediction = predict_digit(data["features"])
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)
