import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train=pd.read_csv('C://Users/Gopi/Desktop/flight/train.csv')
test=pd.read_csv('C://Users/Gopi/Desktop/flight/test.csv')
train.head()
test.head()

print(train.shape)
print(test.shape)

var=train.append(test,sort=False)
var.head()

print(var.shape)
print(var.dtypes)

# Feature Engineering

var['Date']=var['Date_of_Journey'].str.split('/').str[0]
var['Month']=var['Date_of_Journey'].str.split('/').str[1]
var['Year']=var['Date_of_Journey'].str.split('/').str[2]

var['Date']=var['Date'].astype(int)
var['Month']=var['Month'].astype(int)
var['Year']=var['Year'].astype(int)

var=var.drop(['Date_of_Journey'],axis=1)

var['Arrival_Time']=var['Arrival_Time'].str.split(' ').str[0]

var[var['Total_Stops'].isnull()]
var['Total_Stops']=var['Total_Stops'].fillna('1 stop')
var['Total_Stops']=var['Total_Stops'].replace('non-stop','0 stop')

var['Stop'] = var['Total_Stops'].str.split(' ').str[0]

var['Stop']=var['Stop'].astype(int)
var=var.drop(['Total_Stops'],axis=1)

var['Arrival_Hour'] = var['Arrival_Time'] .str.split(':').str[0]
var['Arrival_Minute'] = var['Arrival_Time'] .str.split(':').str[1]

var['Arrival_Hour']=var['Arrival_Hour'].astype(int)
var['Arrival_Minute']=var['Arrival_Minute'].astype(int)
var=var.drop(['Arrival_Time'],axis=1)

var['Departure_Hour'] =var['Dep_Time'] .str.split(':').str[0]
var['Departure_Minute'] =var['Dep_Time'] .str.split(':').str[1]

var['Departure_Hour']=var['Departure_Hour'].astype(int)
var['Departure_Minute']=var['Departure_Minute'].astype(int)
var=var.drop(['Dep_Time'],axis=1)

var['Route_1']=var['Route'].str.split('→ ').str[0]
var['Route_2']=var['Route'].str.split('→ ').str[1]
var['Route_3']=var['Route'].str.split('→ ').str[2]
var['Route_4']=var['Route'].str.split('→ ').str[3]
var['Route_5']=var['Route'].str.split('→ ').str[4]


var['Route_1'].fillna("None",inplace=True)
var['Route_2'].fillna("None",inplace=True)
var['Route_3'].fillna("None",inplace=True)
var['Route_4'].fillna("None",inplace=True)
var['Route_5'].fillna("None",inplace=True)

var=var.drop(['Route'],axis=1)
var=var.drop(['Duration'],axis=1)

var['Price'].fillna((var['Price'].mean()),inplace=True)

from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()

var["Airline"]=encoder.fit_transform(var['Airline'])
var["Source"]=encoder.fit_transform(var['Source'])
var["Destination"]=encoder.fit_transform(var['Destination'])
var["Additional_Info"]=encoder.fit_transform(var['Additional_Info'])
var["Route_1"]=encoder.fit_transform(var['Route_1'])
var["Route_2"]=encoder.fit_transform(var['Route_2'])
var["Route_3"]=encoder.fit_transform(var['Route_3'])
var["Route_4"]=encoder.fit_transform(var['Route_4'])
var["Route_5"]=encoder.fit_transform(var['Route_5'])

print(var.columns)
print(var.dtypes)
var.head()

# FEATURE SELECTION

from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

var_train=var[0:10683]
var_test=var[10683:]

X=var_train.drop(['Price'],axis=1)
y=var_train.Price

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

model=SelectFromModel(Lasso(alpha=0.005,random_state=0))
print(model.fit(X_train,y_train))
print()
print(model.get_support())

print(X.columns)
print()
selected_features=X_train.columns[(model.get_support())]
print(selected_features)

X_train=X_train.drop(['Year'],axis=1)
X_test=X_test.drop(['Year'],axis=1)

print(X_train.shape)
print(X_test.shape)
