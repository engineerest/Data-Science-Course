import  numpy as np

A = np.array([[5,12,6],[-3,0,14]])
print(A)
print(A.T)

B = np.array([[5,3],[-2,4]])
print(B)
print(B.T)

C = np.array([[4,-5],[8,12],[-2,-3],[19,0]])
print(C)
print(C.T)

x = np.array([1,2,3])
print(x)
print(x.T)
print(x.shape)

x_reshaped = x.reshape(1, 3)
print(x_reshaped)
print(x_reshaped.T)