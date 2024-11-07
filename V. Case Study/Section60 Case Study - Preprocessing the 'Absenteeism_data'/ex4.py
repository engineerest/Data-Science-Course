import pandas as pd
from ex3 import df_concatenated

print(df_concatenated.columns.values)
column_names = ['Reason for Absence', 'Date', 'Transportation Expense',
       'Distance to Work', 'Daily Work Load Average', 'Body Mass Index',
       'Education', 'Children', 'Pets', 27,
       28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 43, 46, 47, 48,
       49, 50, 58, 'Absenteeism Time in Hours']

df_concatenated = df_concatenated[column_names]
print(df_concatenated)

# I took.