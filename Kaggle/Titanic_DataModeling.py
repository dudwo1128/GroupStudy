from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
train = pd.read_csv('C:/Users/dudwo/Desktop/ML/Data/titanic/train.csv')
test = pd.read_csv('C:/Users/dudwo/Desktop/ML/Data/titanic/test.csv')
train = train.drop(['Ticket','Cabin'],axis = 1) # Ticket, Cabin 열 삭제
test = test.drop(['Ticket','Cabin'],axis = 1)   # Ticket, Cabin 열 삭제
train = train.fillna({"Embarked" : "S"})
embarked_mapping = {"S":1,"C":2,"Q":3}
train["Embarked"] = train["Embarked"].map(embarked_mapping)
test["Embarked"] = test["Embarked"].map(embarked_mapping)
combine = [train, test]
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.',expand= False)
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(
        ['Lady', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mile', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
title_mapping = {"Mr":1,"Miss":2,"Mrs":3,"Master":4,"Royal":5,"Rare":6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
train = train.drop(['Name','PassengerId'],axis =1)
test = test.drop(['Name','PassengerId'],axis =1)
combine = [train,test]
sex_mapping = {"male":0,"female":1}
for dataset in combine :
    dataset['Sex'] = dataset["Sex"].map(sex_mapping)
train['Age'] = train['Age'].fillna(-0.5)
test['Age'] = test['Age'].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown','Baby','Child','Teenager','Student','Young Adult','Adult','Senior']
train['AgeGroup'] = pd.cut(train["Age"],bins, labels= labels)
test['AgeGroup'] = pd.cut(test["Age"],bins, labels=labels)
age_title_mapping = {1:"Young Adult",2:"Student",3:"Adult",4:"Baby",5:"Adult",6:"Adult",}
for x in range(len(train["AgeGroup"])):
    if train["AgeGroup"][x] == "Unknown" :
        train["AgeGroup"][x] = age_title_mapping[train["Title"][x]]
for x in range(len(test["AgeGroup"])):
    if test["AgeGroup"][x] == "Unknown" :
        test["AgeGroup"][x] = age_title_mapping[test["Title"][x]]
age_mappaing = {"Baby":1,"Child":2,"Teenager":3,"Student":4,"Young Adult":5,"Adult":6,"Senior":7}
train['AgeGroup'] = train['AgeGroup'].map(age_mappaing)
test['AgeGroup'] = test['AgeGroup'].map(age_mappaing)
train = train.drop(['Age'],axis =1)
test = test.drop(['Age'],axis =1)
train['FareBand'] = pd.qcut(train['Fare'], 4, labels= [1,2,3,4])
test['FareBand'] = pd.qcut(train['Fare'],4, labels=[1,2,3,4])
train = train.drop(['Fare'],axis = 1)
test = test.drop(['Fare'],axis=1)


train_data = train.drop('Survived', axis = 1)
target = train['Survived']

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

clf = RandomForestClassifier(n_estimators=13)
clf.fit(train_data,target)

# Model Evaluating
scoring = 'accuracy'
score = cross_val_score(clf,train_data,target,cv=k_fold,n_jobs=1,scoring=scoring)
#print(score)
#print(round(np.mean(score)*100,2))

# test 파일 예측 실행

# test 파일 PassengerId 열 불러오기 위함
test2 = pd.read_csv('C:/Users/dudwo/Desktop/ML/Data/titanic/test.csv')

# 예측
prediction = clf.predict(test)
submission = pd.DataFrame({
    "PassengerId" : test2["PassengerId"],
    "Survived" : prediction
})

print(submission)