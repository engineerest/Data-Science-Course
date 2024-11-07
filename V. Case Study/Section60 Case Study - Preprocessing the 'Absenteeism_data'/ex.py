import pandas as pd

raw_csv_data = pd.read_csv('Absenteeism_data.csv')

df = raw_csv_data.copy()

df_no_age = df.drop(['Age'], axis=1)

print(df_no_age)

# I find out solution, not further in the course