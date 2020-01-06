import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

var=pd.read_csv('C://Users/Gopi/Desktop/santa-2019-revenge-of-the-accountants/data.csv')

print(var.shape)

var.columns

sample=var

y=sample['n_people']

del var['n_people']

del var['family_id']

X=var

print(X.shape)
print(y.shape)

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
tree.fit(X,y)

y_pred=tree.predict(X)
print(y_pred)

pred=pd.DataFrame(y_pred)
sub_df=pd.read_csv('C://Users/Gopi/Desktop/santa-2019-revenge-of-the-accountants/sample_submission.csv')
datasets=pd.concat([sub_df['family_id'],pred],axis=1)
datasets.columns=['family_id','assigned_day']
datasets.to_csv('sample_submission.csv',index=False)