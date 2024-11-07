import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

sns.set()

data = pd.read_csv('1.01.+Simple+linear+regression.csv')

y = data['GPA']
x1 = data['SAT']

plt.scatter(x1,y)
yhat = 0.0017*x1 + 0
fig = plt.plot(x1,yhat, lw=4, color='green', label='regression line')
plt.xlabel('SAT', fontsize=20)
plt.ylabel('GPA', fontsize=20)
plt.xlim(0)
plt.ylim(0)
plt.show()