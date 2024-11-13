#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the datasets
productivity_data = pd.read_csv('C:/Users/fitza/OneDrive/Documents/Canada Productivity Data Final.csv')

# Coefficients from your Ridge regression model
coefficients = {
    'Pop. %': -0.00286228,
    'Working Age Pop.': -0.01034036,
    'Immigration': -0.00449433,
    'Immigration/Pop.': -0.00056041,
    'Capital Formation (Index 1961)': -0.00141432,
    'Capital Formation %': 0.00900890,
    'Minimum Wage': 0.00412808,
    'Manufacturing (Value Added %)': 0.00720018,
    'Services Total Employment (%)': -0.00135705,
    'Inflation (%)': 0.00025048,
    'Prime rate': -0.00696779,
    'US$/CAN': 0.00381476,
    'UKPound/CAN': -0.00610590,
    'Unemployment rate': -0.00816752
}

# Model intercept
intercept = 0.03510344827586208

# Title
st.title("Productivity Forecast Application")

# Load your dataset
data = productivity_data

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

# Prediction calculation
predicted_productivity = intercept + sum(inputs[var] * coef for var, coef in coefficients.items())

if np.isnan(predicted_productivity):
    st.write("Predicted Labour Productivity: Invalid due to missing data.")
else:
    st.write(f"Predicted Labour Productivity for {year}: {predicted_productivity:.2f}")

# Plotting (Fix for TypeError: 'value' must be an instance of str or bytes, not a float)
st.write("Productivity Forecast over Time")
fig, ax = plt.subplots()

# Ensure 'Year' is treated as a numerical value for plotting
ax.plot(data['Year'], data['Labour Productivity'], label='Actual Productivity')

# Ensure the predicted value is plotted correctly
ax.scatter(year, predicted_productivity, color='red', label='User Prediction')

ax.set_xlabel('Year')
ax.set_ylabel('Labour Productivity')
ax.legend()
st.pyplot(fig)


# In[ ]:




