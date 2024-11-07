products = ['Product A', 'Product B']
exp_sales = [10000, 11000, 12000, 13000, 14000]
time_horizon = (1, 3, 12)

for i in products:
    for j in exp_sales:
        for k in time_horizon:
            print([i, j*k])

for prod in products:
    for sale in exp_sales:
        for t_hor in time_horizon:
            print([prod, sale*t_hor])

for prod in products:
    for sale in exp_sales:
        for t_hor in time_horizon:
            print('Expected sales for a period of {0} month(s) for {1}: ${sales}'.format(t_hor, prod, sales=sale*t_hor))