import pandas as pd
import numpy as np
import os
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Step 1: Train the model
def train_model():
    print("Training model, please wait...")
    # Use the compressed data file instead of the original
    data_path = os.path.join("data", "creditcard_reduced.csv")
    df = pd.read_csv(data_path)
    fraud = df[df['Class'] == 1]
    non_fraud = df[df['Class'] == 0].sample(n=fraud.shape[0], random_state=42)
    data = pd.concat([fraud, non_fraud])
    X = data.drop("Class", axis=1)
    y = data["Class"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    print("✅ Model trained.")
    return model

model = train_model()

# Step 2: Flask App
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")

@app.route("/check", methods=["GET"])
def check():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_str = request.form["features"]
        features = [float(x.strip().replace('"', '').replace("'", '')) for x in input_str.split(",")]
        if len(features) != 30:
            raise ValueError("Exactly 30 values are required (V1-V28, Amount, Time).")
        prediction = model.predict([features])[0]
        result = "🚨 Fraud Transaction Detected!" if prediction == 1 else "✅ Normal Transaction"
        return render_template("index.html", prediction=result)
    except Exception as e:
        return render_template("index.html", prediction=f"❌ Error: {e}")

# Step 3: Run app
if __name__ == "__main__":
    print("✅ Flask app is starting...")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))