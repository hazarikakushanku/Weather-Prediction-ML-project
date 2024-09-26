# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Streamlit app starts here
st.title("Rainfall Prediction Model")

# Step 1: Load the dataset
st.sidebar.header("Upload your CSV file")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Step 2: Data Preprocessing
    df.columns = df.columns.str.strip()  # Clean column names

    # Identify non-numeric columns
    non_numeric_cols = df.select_dtypes(include=['object']).columns

    # Encode categorical variables if necessary
    label_encoder = LabelEncoder()
    for col in non_numeric_cols:
        if not df[col].isin([0, 1]).all():  # Only apply encoding if it's not already 0/1
            df[col] = label_encoder.fit_transform(df[col])

    # Handle missing values in numeric columns by filling with the mean
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Step 4: Split the data into train and test sets
    # Replace 'rainfall' with your actual target variable
    X = df.drop('rainfall', axis=1)  # Independent variables
    y = df['rainfall']  # Dependent/Target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 5: Normalize/Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 6: Build and Train the Linear Regression Model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Step 7: Model Evaluation
    y_pred = model.predict(X_test_scaled)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Display model evaluation metrics on the homepage
    st.subheader("Model Evaluation")
    st.write(f"Mean Squared Error: {mse}")
    st.write(f"R-Squared Score: {r2}")

    # Step 8: Visualization of Prediction
    plt.figure(figsize=(8, 4))
    plt.scatter(y_test, y_pred)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
    plt.xlabel('Actual Rainfall')
    plt.ylabel('Predicted Rainfall')
    plt.title('Actual vs Predicted Rainfall')
    plt.savefig('actual_vs_predicted_rainfall.png')  # Save the plot
    plt.close()

    # Display navigation buttons for plot access
    if st.button("Show Boxplots"):
        st.subheader("Boxplot for Numeric Columns")
        for col in numeric_cols:
            plt.figure(figsize=(8, 4))
            plt.boxplot(df[col].dropna())
            plt.title(f'Boxplot for {col}')
            plt.savefig(f'{col}_boxplot.png')
            plt.close()
            st.image(f'{col}_boxplot.png', caption=f'Boxplot for {col}')

    # Show the scatter plot on another page
    if st.button("Show Prediction Plot"):
        st.subheader("Prediction Plot")
        st.image('actual_vs_predicted_rainfall.png', caption="Actual vs Predicted Rainfall")
else:
    st.warning("Please upload a CSV file to proceed.")

