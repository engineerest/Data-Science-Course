# .format() a very useful tool for preprocessing text data
time_horizon = 1, 3, 12
print(time_horizon)

products = ['Product A', 'Product B']
print(products)

# .format() is applicable to string values only

print('Expected sales for a period of {} month(s) for {}:'.format(12, 'Product B'))
# Place holders where we will insert the values that have passed as arguments of the .format() method
print('Expected sales for a period of {} month(s) for {}:'.format(time_horizon[2], products[1]))
print('Expected sales for a period of {0} month(s) for {1}:'.format(time_horizon[2], products[1]))
print('Expected sales for a period of {1} month(s) for {0}:'.format(time_horizon[2], products[1]))
print('Expected sales for a period of {t_hor} month(s) for {prod}:'.format(t_hor=12, prod='Product B'))
print('Expected sales for a period of {t_hor} month(s) for {prod}:'.format(t_hor=12, prod=['Product A', 'Product B']))
print('Expected sales for a period of {t_hor[2]} month(s) for {prod[1]}:'.format(t_hor=time_horizon, prod=products))
print('Expected sales for a period of {t_hor[2]} month(s) for {prod[1]}: ${sales}'.format(t_hor=time_horizon, prod=products, sales=10000))
# keyword arguments = named arguments

