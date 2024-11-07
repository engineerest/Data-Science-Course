import numpy as np
from sklearn.model_selection import train_test_split

a = np.arange(1, 101)
b = np.arange(501, 601)

# Split the data

train_test_split(a)

a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.2, random_state=42)
a_train.shape, a_test.shape
b_train.shape, b_test.shape
print(a_train)
print(a_test)
print(b_train)
print(b_test)

#train_test_split(x,y) splits arrayS or matriceS into random train an test subsets