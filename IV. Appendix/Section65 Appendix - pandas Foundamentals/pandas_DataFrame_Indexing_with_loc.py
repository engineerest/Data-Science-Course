import pandas as pd

data = pd.read_csv('Lending-company.csv', index_col='StringID')
lending_co_data = data.copy()
print(lending_co_data)

# .loc[] has been designed to let you take advantage of the explicit index and column labels of your data table
print(lending_co_data.loc['LoanID_3'])
print(lending_co_data.loc['LoanID_3', :])
print(lending_co_data.loc['LoanID_3', 'Region'])
print(lending_co_data['Location'])
print(lending_co_data.loc['Location'])
print(lending_co_data.loc[:, 'Location'])
print(lending_co_data.loc[:, 'Locations'])