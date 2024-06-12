import pandas as pd
import pickle
import json
from datetime import datetime

# Load the preprocessor and model
with open('E:\salary_prediction\preprocessor\preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

with open('E:\salary_prediction\models\model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('E:\salary_prediction\src\columns.json', 'r') as f:
    columns = json.load(f)
    input_columns = columns['data_columns']
print("Column names loaded successfully:", input_columns)

# Preprocess input data
def preprocess_input(data):
    # Create a DataFrame with the provided data
    df = pd.DataFrame([data], columns=['sex', 'designation', 'age', 'unit', 'leaves_used', 'leaves_remaining', 'ratings', 'past_exp', 'doj', 'current_Date'])
    
    # Extract date of joining and current date
    doj, current_date = data[-2], data[-1]
    # Calculate days in company
    days_in_cmpny = (current_date - doj).days
    
    # Add calculated features: SALARY_AGE, SALARY_EXP, SALARY_DAYS_CMPNY
    df['salary_age'] = df['age'] * 1
    df['salary_exp'] = df['past_exp'] * 1
    df['salary_days_cmpny'] = days_in_cmpny * 1
    
    # Add 'days_in_cmpny' column
    df['days_in_cmpny'] = days_in_cmpny
    
    # Reorder columns to match the expected order and convert to uppercase
    df = df[['sex', 'designation', 'age', 'unit', 'leaves_used', 'leaves_remaining', 'ratings', 'past_exp', 'days_in_cmpny', 'salary_age', 'salary_exp', 'salary_days_cmpny']]

    # Check if the columns in the DataFrame match the expected columns from columns.json
    print("Columns in DataFrame:")
    print(df.columns)
    print("\nColumns loaded from columns.json:")
    print(input_columns)

    assert set(df.columns) == set(input_columns), "Columns in the DataFrame do not match expected columns from columns.json"
    return df

# Predict salary
def predict_salary(data):
    preprocessed_data = preprocess_input(data)
    print("Data after preprocessing function:\n", preprocessed_data)
    
    # Ensure the length of columns after preprocessing matches expected columns
    assert len(preprocessed_data.columns) == len(input_columns), "Number of columns in preprocessed data does not match the expected number of columns."
    
    # Perform preprocessing using the preprocessor pipeline
    preprocessed_data = preprocessor.transform(preprocessed_data)
    print("Data after applying preprocessor:\n", preprocessed_data)
    
    # Make prediction
    prediction = model.predict(preprocessed_data)
    return prediction

# Example usage
example_data = ['Male', 'Data Scientist', 25, 'Engineering', 0, 30, 3.0, 2, datetime(2020, 1, 1), datetime(2024, 6, 11)]
prediction = predict_salary(example_data)
print("Predicted salary:", prediction)
