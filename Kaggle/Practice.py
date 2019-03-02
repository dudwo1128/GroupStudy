import numpy as np
import pandas as pd

train = pd.read_csv('./Data/titanic/train.csv') # train 데이터
test = pd.read_csv('./Data/titanic/test.csv')   # test 데이터

train = train.drop(['Ticket','Cabin'],axis = 1) # Ticket, Cabin 열 삭제
test = test.drop(['Ticket','Cabin'],axis = 1)   # Ticket, Cabin 열 삭제


train = train.fillna({"Embarked" : "S"}) # Embarked에서 s가 가장 많으므로 NaN값을 S로 대체

# Embarked의 S,Q,C를 1,2,3으로 대체
embarked_mapping = {"S":1,"C":2,"Q":3}
train["Embarked"] = train["Embarked"].map(embarked_mapping)
test["Embarked"] = test["Embarked"].map(embarked_mapping)
# print(train.head())

# Name열 문자열 파싱

combine = [train, test]

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.',expand= False)

# Name열 파싱을 Sex열과 함께 나열
#print(pd.crosstab(train['Title'],train['Sex']))

# Name열 구성요소를 6개로 대체

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

# Name, PassengerId 열 삭제

train = train.drop(['Name','PassengerId'],axis =1)
test = test.drop(['Name','PassengerId'],axis =1)
combine = [train,test]

sex_mapping = {"male":0,"female":1}
for dataset in combine :
    dataset['Sex'] = dataset["Sex"].map(sex_mapping)
print(train.head())


# Age 값 정리
train['Age'] = train['Age'].fillna(-0.5)
test['Age'] = test['Age'].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown','Baby','Child','Teenager','Student','Young Adult','Adult','Senior']

# Age 값 정리 : Cut 함수로 각 구간을 득정 값으로 정의
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


# Fare 4개의 범위로 구분

train['FareBand'] = pd.qcut(train['Fare'], 4, labels= [1,2,3,4])
test['FareBand'] = pd.qcut(train['Fare'],4, labels=[1,2,3,4])

train = train.drop(['Fare'],axis = 1)
test = test.drop(['Fare'],axis=1)

print(train.head())