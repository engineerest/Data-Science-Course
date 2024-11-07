# Part 1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn.linear_model import LinearRegression
sns.set()

raw_data = pd.read_csv('1.04.+Real-life+example.csv')
print(raw_data.head())

# PreProcessing

print(raw_data.describe(include='all'))

data = raw_data.drop(['Model'], axis=1) # DataFrame.drop(columns, axis) returns new object with
# the indicated columns dropped
print(data.describe(include='all'))

print(data.isnull().sum())

data_no_mv = data.dropna(axis=0)
print(data_no_mv.describe(include='all'))

sns.histplot(data_no_mv['Price'])
# DataFrame.quantile(the quantile) returns the value at the give quantile (= np.percentile)

# Dealing with outliers

q = data_no_mv['Price'].quantile(0.99)
data_1 = data_no_mv[data_no_mv['Price']<q]
print(data_1.describe(include='all'))

q = sns.displot(data_no_mv['Mileage']).quantile(0.99)
data_2 = data_no_mv[data_1['Mileage']<q]

sns.histplot(data_2['Mileage'])

EngV = pd.DataFrame(raw_data['EngineV'])
EngV.dropna(axis=0)

EngV.sort_values(by='EngineV')

sns.histplot(data_no_mv['EngineV'])

data_3 = data_2[data_2['EngineV']<6.5]
sns.histplot(data_3['EngineV'])

sns.histplot(data_no_mv['Year'])

q = data_3['Year'].quantile(0.01)
data_4 = data_3[data_3['Year']>q]
sns.histplot(data_4['Year'])

data_cleaned = data_4.reset_index(drop=True)
print(data_cleaned.descrive(include='all'))

# Part 2
# Checking the OLS assumptions

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3))
ax1.scatter(data_cleaned['Year'], data_cleaned['Price'])
ax1.set_title('Price and Year')
ax2.scatter(data_cleaned['EngineV'], data_cleaned['Price'])
ax2.set_title('Price and EngineV')
ax3.scatter(data_cleaned['Mileage'], data_cleaned['Price'])
ax3.set_title('Price and Mileage')

plt.show()

sns.histplot(data_cleaned['Price'])

# np.log(x) returns the natural logarithm of a number or array of numbers

log_price = np.log(data_cleaned['Price'])
data_cleaned['log_price'] = log_price
print(data_cleaned)

data_cleaned = data_cleaned.drop(['Price'], axis=1)
# Multicollinearity
print(data_cleaned.columns.values)

from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data_cleaned[['Mileage', 'Year', 'EngineV']]
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif['features'] = variables.columns
print(vif)

data_no_multicollinearity = data_cleaned.drop(['Year'], axis=1)

# Part 3

# Create dummy variables

# pd.get_dummies(df [, drop_first]) spots all categorical variables and creates dummies automatically

data_with_dummies = pd.get_dummies(data_no_multicollinearity, drop_first=True)
print(data_with_dummies.head())

cols = ['log_price', 'Mileage', 'EngineV', 'Brand_BMW',
        'Brand_Mercedes-benz', 'Brand_Mitsubishi', 'Brand_Renault',
        'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',
        'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',
        'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes'
        ]

data_preprocessed = data_with_dummies[cols]
print(data_preprocessed.head())

# Linear Regression linear

targets = data_preprocessed['log_price']
inputs = data_preprocessed.drop(['log_price'], axis=1)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(inputs)

inputs_scaled = scaler.transform(inputs)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.2, random_state=365)

# Part 4

# Create the regression

reg = LinearRegression()
reg.fit(x_train, y_train)

y_hat = reg.predict(x_train)

plt.scatter(y_train, y_hat)
plt.xlabel('Targets (y_train)', size=18)
plt.ylabel('Predictions (y_hat)', size=18)
plt.xlim(6,13)
plt.ylim(6,13)

plt.show()

sns.histplot(y_train - y_hat)
plt.title("Residuals PDF", size=18)
reg.score(x_train, y_train)

# Finding the weights and bias

reg.intercept_

reg.coef_

reg_summary = pd.DataFrame(inputs.olumns.value, columns=['Features'])
reg_summary['Weights'] = reg.coef_
print(reg_summary)

print(data_cleaned['Brand'].unique())

# Part 5

# Testing

y_hat_test = reg.predict(x_test)

plt.scatter(y_test, y_hat_test)
plt.xlabel('Targets (y_test)', size=18, alpha=0.2)
plt.ylabel('Predictions (y_hat_test)', size=18)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()

# plt.scatter(x,y [, alpha]) creates a scatter plot alpha: specifies the opacity

df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Prediction'])
print(df_pf.head())

# np.exp(x) returns the exponential of x (the Euler number 'e' to the power of x)

df_pf['Target'] = np.exp(y_test)
print(df_pf.head())

y_test = y_test.reset_index(drop=True)
print(y_test.heaad())

df_pf['Target'] = np.exp(y_test)
print(df_pf.head())

df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']

df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)

print(df_pf.describe())

pd.options.display.max_rows = 99
pd.set_option('display.float_format', lambda x: '%.2f' % 2)
df_pf.sort_values(by=['Difference%'])
