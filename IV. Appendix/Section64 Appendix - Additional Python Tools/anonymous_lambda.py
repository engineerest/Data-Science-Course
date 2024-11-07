# Lambda Functions in Python

def raise_to_the_power_of_2(x):
    return x**2

print(raise_to_the_power_of_2(3))

# Lambda expressions = Python's syntax for creating anonymous (Lambda) functions

raise_to_the_power_of_2_lambda = lambda x: x**2
print(raise_to_the_power_of_2_lambda(3))

print((lambda x: x/ 2)(11))

sum_xy = lambda x, y: x + y
print(sum_xy(2,3))

sum_xy = lambda x, y: x + y(x)
# print(sum_xy(2, 3)) error
sum_xy = lambda x, y: x + y(x)
# print(sum_xy(2)) error

print(sum_xy(2, lambda x: x*5))
