from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("breast_cancer_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']  
    data_scaled = scaler.transform([np.array(data)])
    prediction = model.predict(data_scaled)
    result = "Malignant" if prediction[0] == 1 else "Benign"
    return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(debug=True)
