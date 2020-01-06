import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

var=pd.read_csv('C://Users/Gopi/Desktop/titanic/test.csv')

var['Age'] = var['Age'].fillna(var['Age'].mean())

var['Cabin'] = var.Cabin.fillna(0)

varun=var
varun.drop(['Name','Ticket','Cabin'],axis=1,inplace = True)

a=pd.get_dummies(varun['Sex'])
b=pd.get_dummies(varun['Embarked'])

varun=pd.concat([varun,a,b],axis='columns')
varun.drop(['Sex','Embarked'],axis=1,inplace = True)
varun.drop(['PassengerId'],axis=1,inplace = True)

#Downloading Data to the local machine
varun.to_csv('testdata.csv',index=False)