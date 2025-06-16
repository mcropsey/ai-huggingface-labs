# explore_csv_data.py

# This script is used to load and explore a CSV dataset using pandas and matplotlib.
# It is often the first step in any data science or machine learning project: understanding your data.

import pandas as pd  # pandas is a powerful data manipulation library
import matplotlib.pyplot as plt  # matplotlib is used for data visualization

# Load the dataset from a CSV file (change the filename as needed)
df = pd.read_csv("sample_data.csv")  # Reads a CSV file into a pandas DataFrame

# Show the first 5 rows of the dataset to understand what kind of data we're dealing with
print("First 5 rows of the dataset:")
print(df.head())

# Print general information including number of rows, columns, and data types
print("\nDataset Info:")
df.info()

# Describe statistical properties like mean, std, min, max of numeric columns
print("\nSummary statistics:")
print(df.describe())

# Check for missing (null) values in the dataset
print("\nMissing values in each column:")
print(df.isnull().sum())

# Plot a histogram for each numeric column to see the distribution of values
df.hist(figsize=(10, 8))  # Histogram is great for visualizing frequency distributions
plt.tight_layout()  # Adjust spacing between plots
plt.show()

# Correlation heatmap to identify relationships between numeric variables
# Correlation is key in understanding which variables are related, useful in feature selection
import seaborn as sns  # seaborn is a statistical data visualization library

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")  # Displays a correlation matrix
plt.title("Feature Correlation Heatmap")
plt.show()
