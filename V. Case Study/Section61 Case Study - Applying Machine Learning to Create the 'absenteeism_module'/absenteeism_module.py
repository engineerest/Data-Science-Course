# Import the relevant libraries
import numpy as np
import pandas as pd

# Load the data

data_preprocessed = pd.read_csv('Absenteeism_preprocessed.csv')
print(data_preprocessed.head())

# Create the targets

print(data_preprocessed['Absenteeism Time in Hours'].median())

# np.where(condition, value if True, value if False) checks if a condition has been satisfied and assigns a value accordingly

targets = np.where(data_preprocessed['Absenteeism Time in Hours'] > 3, 1, 0)
print(targets)

data_preprocessed['Excessive Absenteeism'] = targets
print(data_preprocessed.head())

# A comment on the targets

print(targets.sum() / targets.shape[0])

data_with_targets = data_preprocessed.drop(['Absenteeism Time in Hours'], axis=1)
print(data_with_targets is data_preprocessed)

print(data_with_targets.head())

# Select the inputs for the regression

print(data_with_targets.shape)

# DataFrame.iloc[row indices, column indices] selects (slices) data by position when given rows and columns wanted pandas
print(data_with_targets.iloc[:, 0:14])
print(data_with_targets.iloc[:, :-1])

unscaled_inputs = data_with_targets.iloc[:,:-1]

# Standardize the data

from sklearn.preprocessing import StandardScaler

absenteeism_scaler = StandardScaler()


absenteeism_scaler.fit(unscaled_inputs)


scaled_inputs = absenteeism_scaler.transform(unscaled_inputs)
# new_data_raw = pd.read_csv('new_data.csv')
# new_data_scaled = absenteeism_scaler.transform(new_data_raw)

print(scaled_inputs)
print(scaled_inputs.shape)

# Split the data into train & test and shuffle

# Import the relevant module

from sklearn.model_selection import train_test_split

# Split

# sklearn.mode_selection.train_test_split(inputs, targets) splits arrays or matrices into random train and test subsets

train_test_split(scaled_inputs, targets)

x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, targets, train_size=0.8, shuffle=True, random_state=20)
# Usually, we opt for splits like 90-10 or 80-20 (because we want to train on more data)

# sklearn.mode_selection.train_test_split(inputs, targets, train_size, shuffle=True, random_state) splits arrays
# or matrices into random train and test subsets
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# Logistic regression with sklearn

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Training the model

reg = LogisticRegression()
reg.fit(x_train, y_train)

# sklearn.linear_model.LogisticRegression.score(inputs, targets) returns the mean accuracy on the given test data and labels
reg.score(x_train, y_train)

# Manually check the accuracy

# sklearn.linear_model.LogisticRegression.predict(inputs) predicts class labels (logistic regression outputs) for given input samples

model_outputs = reg.predict(x_train)
print(model_outputs)

print(model_outputs == y_train)
print(np.sum((model_outputs==y_train)))
print(model_outputs.shape[0])
print((model_outputs==y_train)/model_outputs.shape[0])

# Finding the intercept and coefficients

print(reg.intercept_)
print(reg.coef_)
# print(scaled_inputs.columns.values) error!!!

feature_name = unscaled_inputs.columns.values
summary_table = pd.DataFrame(columns=['Feature name'], data=feature_name)
summary_table['Coefficient'] = np.transpose(reg.coef_)
print(summary_table)

summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
summary_table = summary_table.sort_index()
print(summary_table)

# Interpreting te coefficients
summary_table['Odds_ratio'] = np.exp(summary_table.Coefficients)
print(summary_table)

# DataFrame.sort_values(Series, ascending) sorts the values in a data frame with respect to a given column (Series)

summary_table.sort_values('Odds_ratio', ascending=False)

# Testing the model

reg.score(x_test, y_test)

# sklearn.linear_model.LogisticRegression.predict_proba(x) returns the propability estimates for all possible outputs (classes)
predicted_proba = reg.predict_proba(x_test)
print(predicted_proba)
print(predicted_proba.shape)
print(predicted_proba[:, 1])

# Save the model

# pickle [module] is a Python module used to convert a Python object into a character
import pickle

  #file name\        #/write bytes
with open('model', 'wb') as file:
    pickle.dump(reg, file)
      #save/     #\object to be dumped

with open('scale', 'wb') as file:
    pickle.dump(absenteeism_scaler, file)

