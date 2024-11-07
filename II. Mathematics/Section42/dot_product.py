import numpy as np

x = np.array([2,8,-4])
y = np.array([1,-7,3])
print(np.dot(x, y))

u = np.array([0,2,5,8])
v = np.array([20,3,4,-1])
print(np.dot(u, v))
print(np.dot(5,6))
print(np.dot(10, -2))

# Scalar * Vector

print(x)
print(5*x)

# Scalar * Matrix
A = np.array([[5,12,6],[-3,0,14]])
print(A)
print(3*A)

# Example 2
C = np.array([[-12,5,-5,1,6], [6, -2, 0, 0, -3], [10, 2, 0, 8, 0], [9, -4, 8, 3, -6]])
print(C)

D = np.array([[6,-1],[8,-4],[2,-2],[7,4],[-6,-9]])
print(D)