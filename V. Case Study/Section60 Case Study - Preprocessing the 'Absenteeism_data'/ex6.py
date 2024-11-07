
from ex import df
df_reason_mod = df.copy()

print(df_reason_mod.describe())

print(df_reason_mod['Date'][0])
# print(df_reason_mod['Date'][0].month)
list_months = []

print(df_reason_mod.shape)

for i in range(700):
    list_months.append(df_reason_mod['Date'][i])

print(list_months)
print(len(list_months))

df_reason_mod['Month Value'] = list_months
# Ex 1-2
# df_reason_mod['Date'] = [df_reason_mod['Month Value']]
#, 'Day of the Week'
print(df_reason_mod.describe())

# Ex 3

df_reason_date_mod = df_reason_mod.copy()
print(df_reason_date_mod)
# Idk.
# Task had coded wrong. not matter.