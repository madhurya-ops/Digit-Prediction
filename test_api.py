import requests
import json
import numpy as np

# Generate 784 random values
data = {"data": np.random.rand(784).tolist()}

# Send request
url = "http://127.0.0.1:5000/predict"
headers = {"Content-Type": "application/json"}
response = requests.post(url, json=data, headers=headers)

# Print response
print(response.json())
