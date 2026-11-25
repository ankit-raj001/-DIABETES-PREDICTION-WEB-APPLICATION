import pickle
import numpy as np
import streamlit as st

with open('diabetes_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title('Diabetes Prediction Web App') 
st.write ('Enter the details to check the patient has diabetes or not')

pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=0)
glucose = st.number_input('Glucose Level', min_value=0, max_value=200, value=0)
blood_pressure = st.number_input('Blood Pressure Level', min_value=0, max_value=200, value=0)  
skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=0)
insulin = st.number_input('Insulin Level', min_value=0, max_value=900, value=0)
bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=0.0)
dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.0)
age = st.number_input('Age', min_value=0, max_value=120, value=0)

if st.button('Predict'):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.write('The patient has diabetes.')
    else:
        st.write('The patient does not have diabetes.')
