import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('./weatherHistory.csv')

#print(df['Apparent Temperature (C)'], df['Temperature (C)'])
plt.plot(df['Formatted Date'],df['Summary'])
plt.show()