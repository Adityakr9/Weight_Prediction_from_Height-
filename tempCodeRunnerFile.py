# Prediction of wight by using the height 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,mean_squared_error
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

# uploading the dataset 
dataset = pd.read_csv('/Users/adityayadav/Desktop/ DS  Nareshit/All_Projects/Prediction_Height/Height_Weight.csv')
print(dataset.head())

print(dataset.info())
print(dataset.columns.tolist())

check_null=dataset.isnull().sum()
check_null

# Convert the weight into pounds to kg
dataset['weight_kg']=dataset['weight_pounds'] * 0.43592
dataset['weight_kg']

# convert the height into the inches to feet
dataset['height_feet_inches'] = dataset['height_inches'] // 12 + (dataset['height_inches'] % 12)/ 10
dataset['height_feet_inches']

print(dataset['height_feet_inches'])

print(dataset.shape)
print(dataset.describe)

# visualize the data of height_feet_inches into boxplot
sns.boxplot(x=dataset['height_feet_inches'])

# visualize the data of weight_pounds_kg into boxplot
sns.boxplot(x=dataset['weight_kg'])
# let the consider dependent and independent variable 
X=dataset.iloc[:,1].values
y=dataset.iloc[:,0].values
print(X)
print(y)

# Scaling the data into the same range 
scaler= StandardScaler()
X_scale=scaler.fit_transform(X.reshape(-1,1))
y_scale=scaler.fit_transform(y.reshape(-1,1))

print(X_scale)
print(y_scale)

# split the data into 
X_train,X_test, y_train, y_test= train_test_split(X,y, test_size=0.2,random_state=42)

# Reshape training data
X_train = X_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
# Reshape testing data
X_test = X_test.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# create the model. in linearRegression
lr_reg=LinearRegression()
lr_reg.fit(X_train,y_train)

lr_reg_pred= lr_reg.predict(X_test)

lr_reg_pred[:10]

# Create the Decision Tree Regressor
dtr_reg=DecisionTreeRegressor()
dtr_reg.fit(X_train,y_train)

dtr_reg_pred= dtr_reg.predict(X_train)

#find the mean_Sq_error
mean_squared_error(dtr_reg_pred, y_test)


