import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

data = pd.read_csv('C:/Users/dudwo/Desktop/ML/Data/diabetes.csv')
#print(data.describe())

X = data.drop(['Outcome','Index'],axis=1)
y = data['Outcome']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

sc = StandardScaler()
sc.fit(X_train)

X_train = sc.transform(X_train)
X_test = sc.transform(X_test)


mlp = MLPClassifier(hidden_layer_sizes=(30,30,30),max_iter=100)
mlp.fit(X_train,y_train)

predictions = mlp.predict(X_test)

scoring = 'Accurcay'
score = cross_val_score(mlp,X_train,y_train,n_jobs=1,cv=5)
#print(score)

