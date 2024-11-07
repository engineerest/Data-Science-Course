import pandas as pd

from ex import df_no_age
from ex2 import age_dummies

df_concatenated = pd.concat([df_no_age, age_dummies], axis=1)
print(df_concatenated.head())

# So.