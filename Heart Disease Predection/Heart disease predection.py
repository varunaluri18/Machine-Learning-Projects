import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

var=pd.read_csv('C://Users/Gopi/Desktop/heart.csv')
print(var.shape)

var.isnull().mean().head()

var.isnull().values.any()

y = var['target']
X = var.drop(['target'], axis = 1)

from sklearn.model_selection import train_test_split
X_train ,X_test ,y_train ,y_test =train_test_split(X,y,test_size=0.30)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# KNN

# selecting the K value 
import math
print(math.sqrt(len(y_test)))

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=9) 

knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_pred,y_test)
print(cm)
knnac=accuracy_score(y_pred,y_test)
print(knnac)
print()
from sklearn.model_selection import cross_val_score
k=cross_val_score(knn,X,y,cv=10)
print(k)
k.max()

knn_scores = []
for k in range(1,21):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    score=cross_val_score(knn_classifier,X,y,cv=10)
    knn_scores.append(score.max())
knn_scores=pd.DataFrame(knn_scores)
print(knn_scores.max())

# NB

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
print()
print(n.max())

# Decission Tree

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

tree.fit(X_train,y_train)

y_pred=tree.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_pred,y_test)
print(cm)
treeac=accuracy_score(y_pred,y_test)
print()
print(treeac)

from sklearn.model_selection import cross_val_score
tr=cross_val_score(tree,X,y,cv=10)
print()
print(tr)
tr.max()

# Decission Tree
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

forest.fit(X_train,y_train)

y_pred=forest.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_pred,y_test)
print(cm)
print()
forestac=accuracy_score(y_pred,y_test)
print(forestac)

from sklearn.model_selection import cross_val_score
fo=cross_val_score(forest,X,y,cv=10)
print()
print(fo)
fo.max()

print('knn---------------',knnac)
print('NB----------------',nbac)
print('decision tree-----',treeac)
print('random forest-----',forestac)

# After applying the cross_val_scores
print('knn---------------',knn_scores.max())
print('NB----------------',n.max())
print('decision tree-----',tr.max())
print('random forest-----',fo.max())