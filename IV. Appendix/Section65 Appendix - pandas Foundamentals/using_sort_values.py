import pandas as pd

numbers = pd.Series([15, 1000, 23, 45, 444])
print(numbers)

print(numbers.sort_values())
print(numbers.sort_values(ascending=True))
print(numbers.sort_values(ascending=False))

data = pd.read_csv('Location.csv')
location_data = data.copy()
print(location_data.head())

# print(location_data.sort_values()) error
# .sort_values() arranges the values of the object it's been applied to pandas
# The index valuess comply with the object's data that will lead the way
# print(location_data.sort_values(ascending=True)) error
# print(location_data.sort_values(ascending=False)) error

