# Salary Prediction for Data Professions

## Project Overview

The goal of this project is to predict the salaries of individuals in data professions using various machine learning models. This project includes data preprocessing, exploratory data analysis, feature engineering, model training, evaluation, hyperparameter tuning, and deployment of the model as a Streamlit application.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Data Preprocessing](#data-preprocessing)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Feature Engineering](#feature-engineering)
5. [Data Transformation](#data-transformation)
6. [Model Building and Training](#model-building-and-training)
7. [Model Evaluation](#model-evaluation)
8. [Hyperparameter Tuning](#hyperparameter-tuning)
9. [Recommendations](#recommendations)
10. [Deployment](#deployment)
11. [Conclusion](#conclusion)
12. [Repository Structure](#repository-structure)
13. [How to Run](#how-to-run)
14. [Contact](#contact)

## Data Preprocessing

- **Handling Null Values**: Dropped rows with null values in critical columns such as `last_name`, `doj`, `age`, `leaves_used`, `leaves_remaining`, and `ratings`.
- **Data Type Conversion**: Converted `doj` (date of joining) and `current_date` columns to datetime format for further calculations.

## Exploratory Data Analysis (EDA)

- **Outliers Detection**: Detected outliers using Interquartile Range (IQR) and Z-score methods. Visualized distributions with box plots and histograms.

## Feature Engineering

- Dropped unnecessary columns: `first_name`, `last_name`, `doj`, `current_date`.
- Created new features: `days_in_cmpny`, `salary_age`, `salary_exp`, `salary_days_cmpny`.

## Data Transformation

- **Pipeline for Scaling and Encoding**:
  - Numeric Features: Imputation and Standard Scaling.
  - Categorical Features: Imputation and One-Hot Encoding.
  - Created a Column Transformer to preprocess both numeric and categorical features.

## Model Building and Training

- Split the data into training (80%) and testing (20%) sets.
- Trained several models including:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - Gradient Boosting Regressor (XGBoost)

## Model Evaluation

**Model Performance Metrics**

| Model                       | Mean Absolute Error (MAE) | Mean Squared Error (MSE) | R2 Score |
|-----------------------------|---------------------------|---------------------------|----------|
| Linear Regression           | 1007.98                   | 2199214.49                | 1.00     |
| Decision Tree Regressor     | 457.02                    | 2380585.46                | 1.00     |
| Random Forest Regressor     | 437.79                    | 2342351.47                | 1.00     |
| Gradient Boosting Regressor | 705.83                    | 1801327.69                | 1.00     |

## Hyperparameter Tuning

**Tuned Model Performance**

| Model                       | Mean Squared Error (MSE) | R2 Score |
|-----------------------------|--------------------------|----------|
| Decision Tree Regressor     | 6352810.24               | 0.99     |
| Random Forest Regressor     | 2376586.72               | 1.00     |
| Gradient Boosting Regressor | 1252146.27               | 1.00     |

## Recommendations

1. **Feature Engineering Enhancements**
   - Add relevant features: educational background, job location, company size.
   - Explore interaction terms to capture complex relationships.

2. **Advanced Model Techniques**
   - Implement advanced ensemble methods like stacking or blending.
   - Experiment with deep learning models for complex pattern recognition.

3. **Model Interpretability**
   - Utilize SHAP values to interpret model predictions and feature importance.
   - Apply LIME for instance-level model interpretability.

4. **Continuous Model Evaluation**
   - Implement continuous evaluation and retraining pipelines.
   - Establish a feedback loop with stakeholders to refine the model.

## Deployment

- The best model (Gradient Boosting Regressor) and the preprocessor pipeline were saved using pickle.
- Developed a Streamlit application for real-time salary prediction.
- Repository available on GitHub for collaboration and future improvements.

## Conclusion

- Built and evaluated multiple models to predict salaries in data professions.
- The best model was deployed in a Streamlit app.
- The project is available on GitHub for further development and use.

## Repository Structure

salary_prediction/
├── data/
│ ├── Salary Prediction of Data Professions.csv
├── notebooks/
│ ├── salary_prediction.ipynb
├── models/
│ ├── preprocessor.pkl
│ ├── model.pkl
├── app/
│ ├── streamlit_app.py
├── README.md
├── requirements.txt