import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Apply a fix to the statsmodels library
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

raw_data = pd.read_csv('2.02.+Binary+predictors.csv')
print(raw_data)

data = raw_data.copy()
data['Admitted'] = data['Admitted'].map({'Yes':1, 'No':0})
data['Gender'] = data['Gender'].map({'Female':1, 'Male':0})
print(data)

y = data['Admitted']
x1 = data[['SAT', 'Gender']]

x =sm.add_constant(x1)
reg_log = sm.Logit(y, x)
results_log = reg_log.fit()
print(results_log.summary())

print(np.exp(2.0786))
print(np.exp(1.9449))

# Accuracy

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
print(results_log.predict())

np.array(data['Admitted'])
print(results_log.pred_table())

cm_df = pd.DataFrame(results_log.pred_table())
cm_df.columns = ['Predicted 0', 'Predicted 1']
cm_df = cm_df.rename(index={0:'Actual 0',1: 'Actual 1'})
print(cm_df)

# Testing the model and assessing its accuracy

test = pd.read_csv('2.03.+Test+dataset.csv')
print(test)

test['Admitted'] = test['Admitted'].map({'Yes':1, 'No':0})
test['Gender'] = test['Gender'].map({'Female':1, 'Male':0})
print(test)

test_actual = test['Admitted']
test_data = test.drop(['Admitted'], axis=1)
test_data = sm.add_constant(test_data)
#test_data = test_data[x.columns.values]
print(test_data)

def confusion_matrix(data,actual_values,model):

    pred_values = model.predict(data)
    bins=np.array([0,0.5,1])
    cm = np.histogram(actual_values, pred_values, bins=bins)[0]
    accuracy = (cm[0,0]+cm[1,1])/cm.sum()
    return cm, accuracy

cm = confusion_matrix(test_data,test_actual,results_log)
print(cm)

print('Missclassification rate: '+str((1+1)/19))