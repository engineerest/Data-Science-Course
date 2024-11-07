import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('Country+clusters+standardized.csv', index_col='Country')
# pd.read_csv(*.csv,index_col) loads a given CSV file as a data frame

x_scaled = data.copy()
x_scaled = x_scaled.drop(['Language'], axis=1)
print(x_scaled)

sns.clustermap(x_scaled, cmap='mako')


