#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
st.title("Productivity Forecast Application with Multi-Year Adjustment")

# Filter data to allow only years between 1962 and 2019
data = productivity_data[(productivity_data['Year'] >= 1962) & (productivity_data['Year'] <= 2019)]

# Multi-year selection
selected_years = st.multiselect("Select Years for Prediction Adjustment:", sorted(data['Year'].unique()), default=[1962])

# Initialize inputs dictionary to store user input changes across selected years
inputs = {}

# Dictionary to hold predicted productivity results over time for plotting
predicted_productivity_series = data[['Year', 'Labour Productivity (%)']].copy()
predicted_productivity_series['Adjusted Productivity'] = predicted_productivity_series['Labour Productivity (%)']

# Create a set of columns, one for each selected year
cols = st.columns(len(selected_years))

# Process each selected year for adjustments in separate columns
for i, year in enumerate(selected_years):
    with cols[i]:  # Place inputs for each year in its own column
        st.subheader(f"Adjustments for {year}")
        row = data[data['Year'] == year].iloc[0]
        year_inputs = {}

        # Collect input adjustments for each feature in the model
        for variable in coefficients.keys():
            value = row.get(variable, 0)
            adjusted_value = st.number_input(f"{variable} ({year})", value=value)
            year_inputs[variable] = adjusted_value

        # Store inputs for the selected year
        inputs[year] = year_inputs

        # Prediction calculation based on user inputs
        scaled_data = data[coefficients.keys()]
        scaler = StandardScaler().fit(scaled_data)

        # Scale inputs and make prediction for each adjusted year
        adjusted_values = np.array([year_inputs[var] for var in coefficients.keys()]).reshape(1, -1)
        scaled_adjusted_values = scaler.transform(adjusted_values)
        adjusted_predicted_productivity = intercept + sum(
            scaled_adjusted_values[0][i] * coefficients[var] for i, var in enumerate(coefficients)
        )

        # Update predicted series for plotting adjusted trend
        predicted_productivity_series.loc[predicted_productivity_series['Year'] == year, 'Adjusted Productivity'] = adjusted_predicted_productivity

# Plotting the graph of actual vs adjusted productivity trend
st.write("Productivity Forecast Over Time with Adjustments")
fig, ax = plt.subplots()

# Plot original actual productivity data
ax.plot(predicted_productivity_series['Year'], predicted_productivity_series['Labour Productivity (%)'], label='Actual Productivity')

# Plot adjusted productivity trend based on user inputs
ax.plot(predicted_productivity_series['Year'], predicted_productivity_series['Adjusted Productivity'], label='Adjusted Productivity', linestyle='--', color='red')

# Labeling and legend
ax.set_xlabel('Year')
ax.set_ylabel('Labour Productivity (%)')
ax.legend()
st.pyplot(fig)

