import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Apply a fix to the statsmodels library
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

raw_data = pd.read_csv('2.01.+Admittance.csv')

data = raw_data.copy()
data['Admitted'] = data['Admitted'].map({'Yes':1, 'No':0})
print(data)

y = data['Admitted']
x1 = data['SAT']

# Regression

x = sm.add_constant(x1)
reg_log = sm.Logit(y, x)
results_log = reg_log.fit()

# Summary

print(results_log.summary())

x0 = np.ones(168)
reg_log = sm.Logit(y, x0)
results_log = reg_log.fit()

print(results_log.summary())