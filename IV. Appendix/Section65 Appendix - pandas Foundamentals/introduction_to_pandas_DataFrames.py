# A Revision to pandas DataFrames

import pandas as pd
import numpy as np

array_a = np.array([[3, 2, 1], [6, 3, 2]])
print(array_a)

print(pd.DataFrame(array_a))
print(type(pd.DataFrame(array_a)))
# DataFrame - the most important structure i the pandas library

df = pd.DataFrame(array_a, columns=['Column 1', 'Column 2', 'Column 3'])
print(df)

df = pd.DataFrame(array_a, columns=['Column 1', 'Column 2', 'Column 3'], index=['Row 1', 'Row 2'])
print(df)

data = pd.read_csv('Lending-company.csv', index_col='LoanID')
lending_co_data = data.copy()
print(lending_co_data.head())

print(type(lending_co_data))
