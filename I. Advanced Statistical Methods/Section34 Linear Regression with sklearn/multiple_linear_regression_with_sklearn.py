import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.linear_model import LinearRegression

data = pd.read_csv('1.02.+Multiple+linear+regression.csv')
print(data.head())
print(data.describe())

x = data[['SAT', 'Rand 1,2,3']]
y = data['GPA']

reg = LinearRegression()
reg.fit(x,y)

print(reg.coef_)
print(reg.intercept_)

# Calculating the R-squared

reg.score(x, y)
print(reg.score(x, y))

print(x.shape)

r2 = reg.score(x, y)

n = x.shape[0]
p = x.shape[1]

adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
print(adjusted_r2)

#Feature Selection

from sklearn.feature_selection import f_regression

print(f_regression(x,y))

p_values = f_regression(x,y)[1]
print(p_values)

print(p_values.round(3))

#Creating a summary table
reg_summary = pd.DataFrame(data=x.columns.values, columns=['Features'])

reg_summary['Coefficients'] = reg.coef_
reg_summary['p-values'] = p_values.round(3)


# Create the multiple linear regression
x = data[['SAT', 'Rand 1,2,3']]
y = data['GPA']

# Standardization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x) # fit calculate and stores the mean and standard deviation of each feature

x_scaled = scaler.transform(x) # transforms the unscaled inputs using the information contained in the scaler object(feature-wise)

# new_data = pd.read_csv('new_data.csv')
# scaler # contains all standardization info
# new_data_scaled = scaler.transform(new_data)

#Regression with scaled features
reg = LinearRegression()
reg.fit(x_scaled, y)

#Creating a summary table

reg_summary = pd.DataFrame([['Intercept'],['SAT'],['Rand 1,2,3']], columns=['Features'])
reg_summary['Weights'] = reg.intercept_, reg.coef_[0], reg.coef_[1]

# Making predictions with the standardized coefficients (weights)
new_data = pd.DataFrame(data=[[1700,2],[1800,1]], columns=['SAT', 'Rand 1,2,3'])
reg.predict(new_data)

new_data_scaled = scaler.transform(new_data)
print(new_data_scaled)
reg.predict(new_data_scaled)

# What if we removed the 'Random 1,2,3' variable?

reg_simple = LinearRegression()
x_simple_matrix = x_scaled[:,0]
reg.simple.fit(x_simple_matrix,y)
reg_simple.predict(new_data_scaled[:,0].reshape(-1,1))

