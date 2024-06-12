import pandas as pd
import pickle
import json
from datetime import datetime
import streamlit as st

# Load the preprocessor and model
with open('E:\salary_prediction\preprocessor\preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

with open('E:\salary_prediction\models\model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('E:\salary_prediction\src\columns.json', 'r') as f:
    columns = json.load(f)
    input_columns = columns['data_columns']
    st.write("Column names loaded successfully:", input_columns)


# Function to preprocess user input
def preprocess_input(data):
    df = pd.DataFrame([data], columns=['sex', 'designation', 'age', 'unit', 'leaves_used', 'leaves_remaining', 'ratings', 'past_exp', 'doj', 'current_Date'])
    doj, current_date = data[-2], data[-1]
    days_in_cmpny = (current_date - doj).days
    df['salary_age'] = df['age'] * 1
    df['salary_exp'] = df['past_exp'] * 1
    df['salary_days_cmpny'] = days_in_cmpny * 1
    df['days_in_cmpny'] = days_in_cmpny
    df = df[['sex', 'designation', 'age', 'unit', 'leaves_used', 'leaves_remaining', 'ratings', 'past_exp', 'days_in_cmpny', 'salary_age', 'salary_exp', 'salary_days_cmpny']]
    assert set(df.columns) == set(input_columns), "Columns in the DataFrame do not match expected columns from columns.json"
    return df


# Function to predict salary based on preprocessed data
def predict_salary(data):
    preprocessed_data = preprocess_input(data)
    assert len(preprocessed_data.columns) == len(input_columns), "Number of columns in preprocessed data does not match the expected number of columns."
    preprocessed_data = preprocessor.transform(preprocessed_data)
    prediction = model.predict(preprocessed_data)
    return prediction[0]  # Assuming the model returns a single value


# Streamlit app layout
st.title("Salary Prediction App")

# Create input fields for user data
sex = st.selectbox("Select Sex", ["Male", "Female"])
designation = st.selectbox("Select Designation", ['Analyst', 'Senior Analyst', 'Associate', 'Senior Manager','Manager', 'Director'])
age = st.number_input("Enter Age", min_value=18)
unit = st.selectbox("Select Unit", ['Finance', 'IT', 'Marketing', 'Operations', 'Web', 'Management'])
leaves_used = st.number_input("Enter Leaves Used This Year", min_value=0)
leaves_remaining = st.number_input("Enter Leaves Remaining This Year", min_value=0)
ratings = st.number_input("Enter Performance Rating (0-5)", min_value=0.0, max_value=5.0)
past_exp = st.number_input("Enter Years of Past Experience", min_value=0.0)
doj_str = st.text_input("Enter Date of Joining (YYYY-MM-DD)")
current_date_str = st.text_input("Enter Current Date (YYYY-MM-DD)")

# Convert date strings to datetime objects (handle potential errors)
try:
    doj = datetime.strptime(doj_str, "%Y-%m-%d")
    current_date = datetime.strptime(current_date_str, "%Y-%m-%d")
except ValueError:
    st.error("Invalid date format. Please enter dates in YYYY-MM-DD format.")
    st.stop()

# Submit button to trigger prediction
if st.button("Predict Salary"):
    user_data = [sex, designation, age, unit, leaves_used, leaves_remaining, ratings, past_exp, doj, current_date]
    prediction = predict_salary(user_data)
    st.write(f"Predicted Salary: ${prediction:.2f}")
