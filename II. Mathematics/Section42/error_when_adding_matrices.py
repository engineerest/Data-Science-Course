import numpy as np

print(5 + 5)
print(10 - 4)

# Addition of matrices
m1 = np.array([[5,12,6],[-3,0,14]])
print(m1)
m3 = np.array([[5,3], [-2,4]])
print(m3)
# print(m1 + m3) operands could not be broadcast together with shapes (2, 3) (2, 2)

# Exceptions

print(m1)
print(m1 + 1)
print(m1 + np.array([1]))
