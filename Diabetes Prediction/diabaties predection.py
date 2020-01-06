import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

var=pd.read_csv('C://Users/Gopi/Desktop/data.csv')

print(var.shape)

var.isnull().mean().head()

var.isnull().values.any()

diabetes_dummies = {True: 1, False: 0}

var['diabetes'] = var['diabetes'].map(diabetes_dummies)

var.head()

var['diabetes'].value_counts()

X_columns = ['num_preg', 'glucose_conc', 'diastolic_bp', 'insulin', 'bmi', 'diab_pred', 'age', 'skin']
y_columns = ['diabetes']

X = var[X_columns].values
y = var[y_columns].values

X=pd.DataFrame(X)
y=pd.DataFrame(y)

from sklearn.model_selection import train_test_split
X_train ,X_test ,y_train ,y_test =train_test_split(X,y,test_size=0.30)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# K-Nearest Neighbour

# selecting the K value 
import math
print(math.sqrt(len(y_test)))

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=15) 

knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_pred,y_test)
print(cm)
knnac=accuracy_score(y_pred,y_test)
print(knnac)

from sklearn.model_selection import cross_val_score
k=cross_val_score(knn,X,y,cv=10)
print(k)
k.max()

# NB Classifier

from sklearn.naive_bayes import GaussianNB
nb= GaussianNB()

nb.fit(X_train,y_train)

y_pred=nb.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_pred,y_test)
print(cm)
nbac=accuracy_score(y_pred,y_test)
nbac

from sklearn.model_selection import cross_val_score
n=cross_val_score(nb,X,y,cv=10)
print(n.max())

# Decision Tree

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

tree.fit(X_train,y_train)

y_pred=tree.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_pred,y_test)
print(cm)
treeac=accuracy_score(y_pred,y_test)
print(treeac)

from sklearn.model_selection import cross_val_score
tr=cross_val_score(tree,X,y,cv=10)
print(tr)
tr.max()

# Random Forest

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

forest.fit(X_train,y_train)

y_pred=forest.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_pred,y_test)
print(cm)
forestac=accuracy_score(y_pred,y_test)
print(forestac)

from sklearn.model_selection import cross_val_score
fo=cross_val_score(forest,X,y,cv=10)
print(fo)
fo.max()

print('knn---------------',knnac)
print('NB----------------',nbac)
print('decision tree-----',treeac)
print('random forest-----',forestac)

# After applying the cross_val_scores
print('knn---------------',k.max())
print('NB----------------',n.max())
print('decision tree-----',tr.max())
print('random forest-----',fo.max())