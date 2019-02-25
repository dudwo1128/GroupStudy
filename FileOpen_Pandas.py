import pandas as pd
data = pd.read_csv("C:/Users/dudwo/Documents/SideProject_CardUsage/Card_Usage.csv",sep=',',encoding='utf8')
data = data.dropna(how='any')
#if (data['amount'])>50000:
print(type(data['Amount']))