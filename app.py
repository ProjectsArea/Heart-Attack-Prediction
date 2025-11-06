import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# ---------------- Load Model and Preprocessors ----------------
try:
    model = joblib.load("models/heart_attack_model_rf.joblib")
    scaler = joblib.load("models/scaler_rf.joblib")
    label_encoders = joblib.load("models/label_encoders_rf.joblib")
    y_encoder = joblib.load("models/y_encoder_rf.joblib")
    threshold = joblib.load("models/threshold_rf.joblib")
    model_features = model.feature_names_in_

    print("✅ Model, Scaler, Encoders, and Threshold loaded successfully.")
    print("Model features:", model_features)
except Exception as e:
    print("⚠️ Error loading model or encoders:", e)


# ---------------- Home Route ----------------
@app.route('/')
def home():
    return render_template('home.html')

# ---------------- Form Route ----------------
@app.route('/form')
def form():
    return render_template('form.html')


# ---------------- Prediction Route ----------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        age = float(request.form['age'])
        gender = request.form['gender']
        smoking = request.form['smoking']
        alcohol = request.form['alcohol']
        ecg = request.form['ecg']
        spo2 = float(request.form['spo2'])
        bp = request.form['bp']

        # Split blood pressure into systolic/diastolic
        try:
            systolic, diastolic = map(float, bp.split('/'))
        except:
            systolic, diastolic = 120.0, 80.0  # fallback

        # Build DataFrame with correct model features
        df = pd.DataFrame(np.zeros((1, len(model_features))), columns=model_features)

        # Fill in the user data
        df['Age'] = age
        df['Gender'] = gender
        df['Smoking Status'] = smoking
        df['Alcohol Consumption'] = alcohol
        df['ECG Results'] = ecg
        df['Blood Oxygen Levels (SpO2%)'] = spo2
        df['BP_Systolic'] = systolic
        df['BP_Diastolic'] = diastolic

        # Label encode categorical features
        for col in df.columns:
            if col in label_encoders:
                le = label_encoders[col]
                if df[col].iloc[0] not in le.classes_:
                    df[col] = le.transform([le.classes_[0]])
                else:
                    df[col] = le.transform(df[col])

        # Scale numeric features
        df_scaled = scaler.transform(df)

        # Predict probability
        proba = model.predict_proba(df_scaled)[0][1]
        risk = "High Risk" if proba >= threshold else "Low Risk"

        return render_template(
            'result.html',
            result=risk,
            probability=round(proba * 100, 2)
        )

    except Exception as e:
        print("❌ Error during prediction:", e)
        return render_template(
            'result.html',
            result="Error",
            probability=str(e)
        )


# ---------------- Run App ----------------
if __name__ == '__main__':
    app.run(debug=True)
