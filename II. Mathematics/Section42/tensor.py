import numpy as np

m1 = np.array([[5,12,6],[-3,0,14]])
print(m1)
m2 = np.array([[9,8,7],[1,3,-5]])
print(m2)

t = np.array([m1, m2])
print(t)
print(t.shape)

t_manual = np.array([[[5,12,6],[-3,0,14]],[[9,8,7],[1,3,-5]]])
print(t_manual)

# Addition
print(m1 + m2)

# Exercise
m3 = np.array([[5, 3], [-2, 4]])
m4 = np.array([[7, -5], [3, 8]])
print(m3)
print(m4)
print(m3 - m4)

# Adding vectors together

v1 = np.array([1,2,3,4,5])
v2 = np.array([5,4,3,2,1])
print(v1 + v2)
print(v1 - v2)