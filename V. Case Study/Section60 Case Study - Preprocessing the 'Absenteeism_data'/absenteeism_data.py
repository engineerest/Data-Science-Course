import pandas as pd
from ex6 import df_reason_date_mod

raw_csv_data = pd.read_csv('Absenteeism_data.csv')
print(raw_csv_data)
print(raw_csv_data.head())
print(raw_csv_data.describe())

df = raw_csv_data.copy()
print(df)

# pd.options.display.max_columns = #
pd.options.display.max_columns = None
# pd.options.display.max_rows = #
pd.options.display.max_rows = None

print(df.info())

# label variable
# a number that is there to distinguish the individuals from one another, not to carry any numeric information (nominal data)

# Drop 'ID'
print(df.drop(['ID'], axis=1))
print(df)
print(raw_csv_data)

# Reason for Absence

print(df['Reason for Absence'])
print(df['Reason for Absence'].min())
print(df['Reason for Absence'].max())

print(pd.unique(df['Reason for Absence']))
print(df['Reason for Absence'].unique())
print(len(df['Reason for Absence'].unique()))
print(sorted(df['Reason for Absence'].unique()))
# sorted() returns a new, sorted list from the items in its argument

# dummy variables
# an explanatory binary variable that equals
# 1 if a certain categorical effect is present, and that equals
# 0 if that same effect is absent

# .get_dummies() converts categorical variable into dummy variables

reason_columns = pd.get_dummies(df['Reason for Absence'])
print(reason_columns)
reason_columns['check'] = reason_columns.sum(axis=1)
print(reason_columns)
print(reason_columns['check'].sum(axis=0))
print(reason_columns['check'].unique())
print(reason_columns['check'].unique())
reason_columns = reason_columns.drop(['check'], axis=1)
print(reason_columns)

reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first=True)
print(reason_columns)

# Group the Reasons for Absence

print(df.columns.values)
print(reason_columns.columns.values)

# Group these variables
# re-organizing a certain type of variables into groups in a regression analysis
# Group = Class

df = df.drop(['Reason for Absence'], axis=1)
print(df)
reason_columns.loc[:, 1:14].max(axis=1)
# .loc[] is label-based

reason_type1 = reason_columns.loc[:, 1:14].max(axis=1)
reason_type2 = reason_columns.loc[:, 15:17].max(axis=1)
reason_type3 = reason_columns.loc[:, 18:21].max(axis=1)
reason_type4 = reason_columns.loc[:, 22:].max(axis=1)

# Concatenate Column Values
print(df)

df = pd.concat([df, reason_type1, reason_type2, reason_type3, reason_type4], axis=1)
# pd.concat() = "concatenate"
print(df)
print(df.columns.values)
column_names = ['Date', 'ID','Transportation Expense', 'Distance to Work', 'Age',
           'Daily Work Load Average', 'Body Mass Index', 'Education',
           'Children', 'Pets', 'Absenteeism Time in Hours', 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']

df.columns = column_names
print(df.head())
# .head() displays the top five rows of our data table, together with the relevant column names

# Recorder Columns

column_names_reordered = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4',
                          'Date', 'ID','Transportation Expense', 'Distance to Work', 'Age',
                         'Daily Work Load Average', 'Body Mass Index', 'Education',
                        'Children', 'Pets', 'Absenteeism Time in Hours']

df = df[column_names_reordered]
print(df.head())

# Create a Checkpoint

# Checkpoint
# an interim save of your work

df_reason_mod = df.copy()
print(df_reason_mod)

# In programming in general, and in Jupyter (Project established on PyCharm) in particular, creating checkpoints refers
#to storing the current version of your code, not really the content of a variable

# 'Date':

print(df_reason_mod['Date'])
print(type(df_reason_mod['Date']))
print(type(df_reason_mod['Date'][0]))

df_reason_mod['Date'] = pd.to_datetime(df_reason_mod['Date'])
# pd.to_datetime() converts values into timestamp

print(df_reason_mod['Date'])

df_reason_mod['Date'] = pd.to_datetime(df_reason_mod['Date'], format='%d/%m/%Y')
# %d day
# %m month
# %Y year
# %H hour
# %M minute
# %S second
print(df_reason_mod['Date'])
print(type(df_reason_mod['Date'][0]))
print(df_reason_mod.info())

# Extract thed Month Value
print(df_reason_mod['Date'][0])
print(df_reason_mod['Date'][0].month)
list_months = []

print(df_reason_mod.shape)

for i in range(700):
    list_months.append(df_reason_mod['Date'][i].month)

print(list_months)
print(len(list_months))
df_reason_mod['Month Value'] = list_months
print(df_reason_mod.head(20))

# Extract the Day of the Week

print(df_reason_mod['Date'][699].weekday())
# .weekday() returns an integer corresponding to the day of the week

print(df_reason_mod['Date'][699])

# To apply a certain type of modification iteratively on each value from a Series or a column in a DataFrame, it is a great idea
#to create a function that can execute this operation for one element, and then implement it to all values from the column of interest.

def date_to_weekday(date_value):
    return date_value.weekday()

df_reason_mod['Day of the Week'] = df_reason_mod['Date'].apply(date_to_weekday)
print(df_reason_mod.head())


print(type(df_reason_date_mod['Transportation Expense'][0]))
print(type(df_reason_date_mod['Distance to Work'][0]))
print(type(df_reason_date_mod['Age'][0]))
print(type(df_reason_date_mod['Daily Work Load Average'][0]))
print(type(df_reason_date_mod['Body Mass Index'][0]))

# 'Education', 'Children', 'Pets'
print(df_reason_date_mod)

print(df_reason_date_mod['Education'].unique())
print(df_reason_date_mod['Education'].value_counts())
df_reason_date_mod['Education'] = df_reason_date_mod['Education'].map({1:0, 2:1, 3:1, 4:1})
print(df_reason_date_mod['Education'].unique())

# Final Checkpoint

df_preprocessed = df_reason_date_mod.copy()
print(df_preprocessed.head(10))
df_preprocessed.to_csv('Absenteeism_preprocessed.csv', index=False)