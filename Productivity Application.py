import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the dataset
productivity_data = pd.read_csv('C:/Users/fitza/OneDrive/Documents/Canada Productivity Data Final.csv')

# Coefficients from your Ridge regression model
coefficients = {
    'Pop. %': -0.2509223,
    'Immigration/Working Age Pop. (%)': -0.36431529,
    'Capital Formation (Index 1961)': -0.63939989,
    'Capital Formation %': 0.88167326,
    'Minimum Wage': 0.24379902,
    'Manufacturing (Value Added %)': 1.02497024,
    'Services Total Employment (%)': -0.38547857,
    'Inflation (%)': -0.00899914,
    'Prime rate': -0.69016519,
    'US$/CAN': 0.27471739,
    'UKPound/CAN': -0.60009435,
    'Unemployment rate (%)': -0.90196938
}

# Model intercept
intercept = 3.5103448275862075

# Title
st.title("Productivity Forecast Application")

# Filter data to allow only years between 1962 and 2019
data = productivity_data[(productivity_data['Year'] >= 1962) & (productivity_data['Year'] <= 2019)]

# Select a Year
year = st.selectbox("Select Year for Prediction:", data['Year'].unique())

# Get default values for selected year
row = data[data['Year'] == year].iloc[0]

# Dictionary to store user inputs
inputs = {}

# Check if any required variable is missing and handle missing values
for variable in coefficients.keys():
    if variable in row.index:
        value = row[variable]
        # Handle missing values by defaulting to 0 or using available value
        if pd.isnull(value):
            st.warning(f"Missing value for {variable}, using default value.")
            value = 0
        inputs[variable] = st.number_input(variable, value=value)
    else:
        # Use default value if the column is missing
        inputs[variable] = st.number_input(variable, value=0)

# Standardize the features for prediction
scaler = StandardScaler()

# Extract the features for scaling (from the selected year data)
scaled_data = scaler.fit_transform(data[coefficients.keys()])

# Prediction calculation (scaled inputs)
scaled_input = np.array([inputs[var] for var in coefficients.keys()]).reshape(1, -1)
scaled_input = scaler.transform(scaled_input)  # Standardize the input based on the model's scaling

# Prediction using scaled inputs
scaled_predicted_productivity = intercept + sum(scaled_input[0][i] * coefficients[variable] for i, variable in enumerate(coefficients))

# Inverse scaling for the prediction to return to original scale
predicted_productivity = scaled_predicted_productivity  # No need for inverse transform as you're using standardized prediction

if np.isnan(predicted_productivity):
    st.write("Predicted Labour Productivity: Invalid due to missing data.")
else:
    st.write(f"Predicted Labour Productivity for {year}: {predicted_productivity:.2f}")

# Plotting the graph of predicted vs actual data
st.write("Productivity Forecast over Time")
fig, ax = plt.subplots()

# Plot the actual data for 'Labour Productivity (%)'
ax.plot(data['Year'], data['Labour Productivity (%)'], label='Actual Productivity')

# Plot the predicted value
ax.scatter(year, predicted_productivity, color='red', label='User Prediction')

ax.set_xlabel('Year')
ax.set_ylabel('Labour Productivity (%)')
ax.legend()
st.pyplot(fig)



