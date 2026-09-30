import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('./hotel_bookings.csv')

# -------------------
#initial analysis

# print(df.head())
# print(df.info())

# -------------------
#analysis of null values

print('Dados nulos:')
print(df.isnull().sum())
print('-'*50)

#Conclusion: agent and company has a bunch of null values
#company, counrry, agent need changes

# -------------------
#heatmap analysis

fig, ax = plt.subplots()
numeric_cols = df.select_dtypes(include=[np.number]).columns
sns.heatmap(df[numeric_cols].corr(), cmap='coolwarm')
plt.show()

#conclusion: the variable is_canceled has a common correlation with most of other variables. 
#It has an above-normal correlation with `lead_time` and a below-normal correlation with total_special-requests
