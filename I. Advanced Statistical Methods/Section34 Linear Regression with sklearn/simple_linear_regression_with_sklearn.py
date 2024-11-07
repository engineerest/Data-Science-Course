import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.linear_model import LinearRegression

data = pd.read_csv('1.01.+Simple+linear+regression.csv')
data.head()

x = data['SAT']
y = data['GPA']

x_matrix = x.values.reshape(-1, 1) # we reshape x because we got the error

reg = LinearRegression()
reg.fit(x_matrix,y) # x_matrix is input, y is target

print(reg.score(x_matrix,y)) # return the R-squared
print(reg.coef_) # return coef
print(reg.intercept_) # return const coef

# A simple linear regression always has a single intercept

# print(reg.predict(x_matrix)) return the predictions of the linear regression model for some new inputs

new_data = pd.DataFrame(data=[1740, 1760], columns=['SAT'])

print(reg.predict(new_data))

new_data['Predicted_GPA'] = reg.predict(new_data)
print(new_data)

plt.scatter(x,y)
yhat = reg.coef_*x_matrix + reg.intercept_
fig = plt.plot(x, yhat, lw=4, c="orange", label='regression line')
plt.ylabel('GPA', fontsize=20)
plt.xlabel('SAT', fontsize=20)
plt.show()

