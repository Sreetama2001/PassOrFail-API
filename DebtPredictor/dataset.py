import numpy as np
import pandas as pd 
df= pd.read_csv("DebtPredictor\debt_data.csv")
# Display basic information about the dataset
print(df.info())
# Display statistical summary of numerical columns
print(df.describe())
# Check the first few rows of the dataset
print(df.head())
