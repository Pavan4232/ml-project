import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

df=pd.read_csv(r"C:\Users\pavan\OneDrive\Desktop\dsa\project\traning.py")
# Drop unnecessary column if exists
df = df.drop(columns=["Unnamed: 133"], errors='ignore')

print(df.head())

X = df.drop("prognosis", axis=1)
y = df["prognosis"]

# Encode disease labels
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))

svm_model = SVC()
svm_model.fit(X_train, y_train)

svm_pred = svm_model.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))


def predict_disease(symptoms_input):
    input_data = [0] * len(X.columns)

    for symptom in symptoms_input:
        if symptom in X.columns:
            index = X.columns.get_loc(symptom)
            input_data[index] = 1

    input_data = [input_data]

    prediction = rf_model.predict(input_data)
    disease = le.inverse_transform(prediction)

    return disease[0]



sample_symptoms = ["itching", "skin_rash", "nodal_skin_eruptions"]

result = predict_disease(sample_symptoms)
print("Predicted Disease:", result)

import streamlit as st

st.title("🏥 Disease Prediction System")

symptoms = st.multiselect("Select Symptoms", X.columns)

if st.button("Predict"):
    result = predict_disease(symptoms)
    st.success(f"Predicted Disease: {result}")