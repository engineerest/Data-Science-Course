import pandas as pd

raw_csv_data = pd.read_csv('Absenteeism_data.csv')
df = raw_csv_data.copy()

age_dummies = pd.get_dummies(df['Age'])
print(age_dummies)

# Also.