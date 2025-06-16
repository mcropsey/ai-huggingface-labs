import pandas as pd
import matplotlib.pyplot as plt

# Example CSV from seaborn
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

print(df.head())
df.groupby("species").mean(numeric_only=True).plot(kind="bar")
plt.title("Average Measurements per Iris Species")
plt.ylabel("Measurement (cm)")
plt.grid(True)
plt.tight_layout()
plt.show()
