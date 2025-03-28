import joblib
import numpy as np

# Load trained SVM model using joblib
model_path = "svm_model.pkl"  # Update with your model's actual path
model = joblib.load(model_path)

def predict(data):
    """Accepts input data and returns SVM model prediction."""
    input_data = np.array(data).reshape(1, -1)  # Ensure correct shape
    prediction = model.predict(input_data)[0]
    return int(prediction)  # Convert NumPy int to Python int for JSON response
