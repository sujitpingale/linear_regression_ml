# from flask import Flask, request, render_template
# import pickle
#
# app = Flask(__name__)
#
# # Load the trained model
# with open('model.pkl', 'rb') as file:
#     model = pickle.load(file)
#
#
# @app.route('/')
# def home():
#     return render_template('index.html')
#
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         # Get the input from the form
#         age = int(request.form['age'])
#         income = int(request.form['income'])
#
#         # Predict using the model
#         prediction = model.predict([[age, income]])[0]
#
#         # Return the result
#         if prediction == 1:
#             return render_template('index.html', prediction_text="You are likely to buy the product!")
#         else:
#             return render_template('index.html', prediction_text="You are unlikely to buy the product.")
#
#
# if __name__ == "__main__":
#     app.run(debug=True)

import streamlit as st
import pickle
import numpy as np
import os

# Get the current working directory
model_path = os.path.join(os.getcwd(), 'model.pkl')

# Load the machine learning model
try:
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    st.error("Error: The model.pkl file was not found. Please make sure it is in the same directory as this script.")
    st.stop()

# App title and description
st.title("Machine Learning Prediction App")
st.write("This app uses a pre-trained machine learning model to predict outcomes based on user input.")

# User input
st.header("Enter the input features:")

# Example: Accepting two features as input
feature_1 = st.number_input("Feature 1", value=0.0, format="%.2f")
feature_2 = st.number_input("Feature 2", value=0.0, format="%.2f")

# Button to trigger prediction
if st.button("Predict"):
    # Convert inputs to a numpy array
    input_features = np.array([[feature_1, feature_2]])

    # Make prediction
    prediction = model.predict(input_features)

    # Display prediction
    st.success(f"The predicted result is: {prediction[0]}")

# Footer
st.write("---")
st.write("Powered by Streamlit | Your ML App Made Easy")
