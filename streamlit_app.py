import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="titanic_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load scaler
scaler = joblib.load("scaler.pkl")

# UI
st.set_page_config(page_title="Titanic Survival Prediction 🚢", layout="centered")

st.title("🚢 Titanic Survival Prediction App")
st.subheader("Input your details and find out if you'd survive the Titanic disaster.")


Pclass = st.selectbox("Passenger Class (1 = First, 2 = Second, 3 = Third)", [1, 2, 3])
Sex = st.selectbox("Sex", ["male", "female"])
Age = st.slider("Age", 0, 80, 25)
SibSp = st.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
Parch = st.number_input("Number of Parents/Children Aboard", 0, 10, 0)
Fare = st.slider("Fare", 0.0, 500.0, 50.0)
Embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Feature engineering
Sex = 0 if Sex == "male" else 1
Embarked_Q = 1 if Embarked == "Q" else 0
Embarked_S = 1 if Embarked == "S" else 0
FamilySize = SibSp + Parch + 1
IsAlone = 1 if FamilySize == 1 else 0

# Scale age & fare
scaled = scaler.transform([[Age, Fare]])
Age_scaled, Fare_scaled = scaled[0]

# Final input
input_data = np.array([[Pclass, Sex, Age_scaled, SibSp, Parch, Fare_scaled,
                        Embarked_Q, Embarked_S, FamilySize, IsAlone]], dtype=np.float32)

# Prediction
if st.button("Predict"):
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    prediction = output[0][0]

    st.write(f"Predicted Probability: {prediction:.4f}")
    if prediction >= 0.5:
        st.success("Passenger is likely to **Survive** 🛟")
    else:
        st.error("Passenger is likely to **Not Survive** ❌")
