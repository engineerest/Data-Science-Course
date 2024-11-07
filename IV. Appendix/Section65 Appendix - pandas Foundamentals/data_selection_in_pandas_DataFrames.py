import pandas as pd

data = pd.read_csv('Lending-company.csv', index_col='StringID')
lending_co_data = data.copy()
print(lending_co_data.head())

print(lending_co_data.Product)
print(lending_co_data.Location)

print(lending_co_data['Product'])
print(lending_co_data['Location'])
# print(lending_co_data['location']) error
print(type(lending_co_data['Location']))
print(lending_co_data[['Location']])
print(lending_co_data[['Location', 'Product']].head())
prod_loc = ['Location', 'Product']
print(lending_co_data[prod_loc].head())
# print(lending_co_data['Product', 'Location']) error

# Nesting a list within the indexing operator is most needed when we want to extract data from several columns
# A more elegant way to obtain the same result is to store the list containing the column names in a separate variable

