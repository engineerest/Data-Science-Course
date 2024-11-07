import numpy as np

s = 5
print(s)

# Vectors
v = np.array([5, -2, 4])
print(v)

# Matrices
m = np.array([[5,12,6], [-3,0,14]])
print(m)

# Data types
print(type(s))
print(type(v))
print(type(m))


s_array = np.array(5)
print(s_array)
print(type(s_array))

# shape returns the shape (dimensions) of a variable

# Data shapes
print(m.shape)
print(v.shape)
# print(s.shape) int object has not attribute shape

# reshape gives an array a new shape, without changing its data

# Creating a column vector
print(v.reshape(1,3))
print(v.reshape(3,1))