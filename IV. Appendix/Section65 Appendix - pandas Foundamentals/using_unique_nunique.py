import pandas as pd

data = pd.read_csv('Location.csv')
location_data = data.copy()
print(location_data.head())

# The pandas Series object represents a single-column data or a set of observations related to a single variable
# One-dimensional NumPy array structure

print(location_data)
print(location_data.describe())
print(len(location_data))
print(location_data.nunique())
print(type(location_data.nunique()))
# print(location_data.unique()) error
# The .unique() method delivers the values in the order they have appeared in the data set
# print(type(location_data.unique())) error
# print(location_data.unique) error

# However, attribute unique() has errors