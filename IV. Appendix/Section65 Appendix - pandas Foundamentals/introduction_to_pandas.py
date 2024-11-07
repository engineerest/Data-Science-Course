import pandas as pd

products = ['A', 'B', 'C', 'D']
print(products)
print(type(products))

product_categories = pd.Series(products)
# 0    A
# 1    B
# 2    C
# 3    D
# dtype: object
# object - The default datatype assigned to data whch is not numeric
print(product_categories)
print(type(product_categories))

print(type(pd.Series(products)))
daily_rates_dollars = pd.Series([40, 45, 50, 60])
print(daily_rates_dollars)
# Pandas Series object corresponds to the one-dimensional NumPy array structure

import numpy as np

array_a = np.array([10, 20, 30, 40, 50])
print(array_a)
print(type(array_a))

series_a = pd.Series(array_a)
print(series_a)
print(type(series_a))

# Takeaways: #1: The pandas Series object is something like a powerful version of the Python list,
#or an enhanced version of the NumPy array
# #2: Remember to always maintain data consistency

