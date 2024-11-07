for i in range(2):
    print(i)

for j in range(5):
    print(j)

for i in range(2):
    for j in range(5):
        print(j)

for i in range(2):
    for j in range(5):
        print([i, j])

for i in ['Product A', 'Product B']:
    for j in range(5):
        print([i, j])

products = ['Product A', 'Product B']
exp_sales = [10000, 11000, 12000, 13000, 14000]

for i in products:
    for j in exp_sales:
        print([i, j])