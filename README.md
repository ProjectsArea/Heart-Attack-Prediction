# 🫀 Heart Attack Risk Prediction (AI + IoT Integration)

## 🔍 Overview
This project is a **machine learning–powered web application** that predicts the **risk of heart attack** based on several health parameters such as **age, gender, smoking habits, ECG results, SpO₂ levels, and blood pressure**.

It is built using **Flask (Python)** as the backend and a **Random Forest Classifier** trained in **Google Colab** for accurate heart attack risk prediction.  
The app can also be connected with **IoT devices** (ESP32/ESP8266 with sensors) to collect and analyze real-time health data.

---

## 🚀 Features
- 🧠 Predicts **High or Low Heart Attack Risk**
- 📊 Shows **Probability of High Risk**
- 💻 **Flask web app interface**
- 🔌 **IoT Integration** for real-time sensor data
- ⚙️ Modular and easily extendable
- ☁️ Ready for cloud deployment

---

## 🧱 Project Structure

heart_attack_app/
│
├── app.py # Flask app (main backend)
│
├── models/ # Pretrained model & supporting files
│ ├── heart_attack_model_rf.joblib
│ ├── scaler_rf.joblib
│ ├── label_encoders_rf.joblib
│ ├── y_encoder_rf.joblib
│ └── threshold_rf.joblib
│
├── templates/ # Frontend HTML templates
│ ├── index.html # Input form
│ └── result.html # Result page
│
├── static/ # Static assets
│ ├── css/
│ │ └── style.css # Custom styles
│ └── js/
│ └── script.js # (Optional) JS logic


---

## 🧠 Model Information
- **Algorithm:** Random Forest Classifier  
- **Environment:** Trained in Google Colab  
- **Accuracy:** ~85%  
- **Input Features:**
  - Age  
  - Gender  
  - Smoking Status  
  - Alcohol Consumption  
  - ECG Results  
  - Blood Oxygen Levels (SpO₂)  
  - BP Systolic  
  - BP Diastolic  

- **Output:**
  - Risk Level → *High / Low*  
  - Probability → *Model confidence (%)*

---
