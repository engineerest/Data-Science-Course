# Using Methods in Python - Part 1
import pandas as pd

start_date_deposits = pd.Series({
    '7/4/2014' : 2000,
    '1/2/2015' : 2000,
    '12/8/2012' : 1000,
})
print(start_date_deposits)
# A Python object is associated with a certain collection of attributes and methods

# Functions
# - an independent entity
# Methods
# - can have access to the object's data
# - can manipulate the object's state
# pandas, pandas Series, python, NumPy

print(start_date_deposits.sum) # No value
print(start_date_deposits.sum()) # sum
print(start_date_deposits.min()) # max
print(start_date_deposits.max()) # min
print(start_date_deposits.idxmax()) # idxmax() - returns the index label corresponding to the highest value in a series
print(start_date_deposits.idxmin()) # idxmin() - delivers the start date of the deposit with the smallest value

# Using Methods in Python - Part 2
import pandas as pd
# Pandas is a library, which steps on the computational abilities of NumPy

start_date_deposits = pd.Series({
    '7/4/2014' : 2000,
    '1/2/2015' : 2000,
    '12/8/2012' : 1000,
})
# Numeric data only - NumPy
# both numeric and non-numeric data - pandas

print(start_date_deposits.head()) # The .head() method provides a quick and efficient way for you to catch a glimpse of the structure
#of your dataset
print(start_date_deposits.tail())

# One of the best features of using methods is that we can also modify their performance
print(start_date_deposits.head())
print(start_date_deposits.head(3))
print(start_date_deposits.head(10))
# The .head() method provides us with the option to choose the number of displayed rows from the object it has been applied to

print(start_date_deposits.head())
# This parameter of the .head() method allows us to modify the way in which the method will operate

print(start_date_deposits.head(n=10)) # n=10 - .head(10)

# Pandas methods have parameters you can supply with arguments to modify the performance of the given method
