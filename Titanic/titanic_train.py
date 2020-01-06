import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as snb

var=pd.read_csv('C://Users/Gopi/Desktop/titanic/train.csv')

var['Age'] = var['Age'].fillna(var['Age'].mean())

var['Cabin'] = var.Cabin.fillna(0)


varun=var
varun.drop(['Name','Ticket','Cabin'],axis=1,inplace = True)

a=pd.get_dummies(varun['Sex'])
b=pd.get_dummies(varun['Embarked'])

varun=pd.concat([varun,a,b],axis='columns')

varun.drop(['Sex','Embarked'],axis=1,inplace = True)

varun.drop(['PassengerId','Survived'],axis=1,inplace = True)


varun.head()

q=var.Survived

train_df=pd.concat([varun,q],axis='columns')

train_df.head()

test_df=pd.read_csv('testdata.csv')

final_df=pd.concat([train_df,test_df],axis=0)

print(train_df.shape)
print(test_df.shape)
print(final_df.shape)

final_df =final_df.loc[:,~final_df.columns.duplicated()]

print(final_df.shape)

df_Train=final_df.iloc[:891,:]
df_Test=final_df.iloc[891:,:]

print(df_Train.shape)
print(df_Test.shape)

df_Test.drop(['Survived'],axis=1,inplace=True)

X_train=df_Train.drop(['Survived'],axis=1)
y_train=df_Train['Survived']





# MODEL BUILDING

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
poi=tree.fit(X_train,y_train)

df_Test['Fare']=df_Test['Fare'].fillna(df_Test['Fare'].mean())

y_pred=tree.predict(df_Test)
print(y_pred)

#Submittting the model
pred=pd.DataFrame(y_pred)
sub_df=pd.read_csv('C://Users/Gopi/Desktop/titanic/final_sub.csv')
datasets=pd.concat([sub_df['PassengerId'],pred],axis=1)
datasets.columns=['PassengerId','Survived']
datasets.to_csv('sample_submission.csv',index=False)