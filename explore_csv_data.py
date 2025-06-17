# Import the pandas library, which is used for data manipulation and analysis
import pandas as pd

# Import matplotlib's pyplot module for plotting graphs
import matplotlib.pyplot as plt

# Define the URL of the CSV file (public iris dataset from the seaborn GitHub repository)
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"

# Load the CSV data into a pandas DataFrame
# The CSV contains data about iris flowers, including sepal/petal measurements and species
df = pd.read_csv(url)

# Print the first 5 rows of the dataset to visually inspect the structure of the data
print(df.head())

# Group the data by the 'species' column and calculate the mean of the numeric columns
# This gives us average measurements (sepal length, sepal width, petal length, petal width) per species
# numeric_only=True ensures only numeric columns are used in the mean calculation
df.groupby("species").mean(numeric_only=True).plot(kind="bar")

# Add a title to the plot
plt.title("Average Measurements per Iris Species")

# Label the Y-axis to indicate measurement units
plt.ylabel("Measurement (cm)")

# Display a grid on the plot for better readability
plt.grid(True)

# Automatically adjust subplot parameters to fit the plot neatly in the window
plt.tight_layout()

# Display the bar chart
plt.show()