# Prediction of wight by using the height 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
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

# create the model in linearRegression
lr_reg=LinearRegression()
lr_reg.fit(X_train,y_train)

lr_reg_pred= lr_reg.predict(X_test)

lr_reg_pred[:10]

# Create the Decision Tree Regressor
dtr_reg=DecisionTreeRegressor()
dtr_reg.fit(X_train,y_train)

dtr_reg_pred= dtr_reg.predict(X_test)

#find the mean_Sq_error
mean_squared_error(dtr_reg_pred, y_test)

# Random forest regressiom
rgr=RandomForestRegressor()
rgr.fit(X_train, y_train)

rgr_pred= rgr.predict(X_test)
rgr_pred[:5]

mean_squared_error(rgr_pred, y_test)

# HyperParameter Tuning
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression

param_grid = {
    'fit_intercept' : [True, False],
    'copy_X':[True,False]
}

model_lr = LinearRegression()
grid_search = GridSearchCV(model_lr, param_grid, cv = 5, scoring = 'neg_mean_squared_error')
grid_search.fit(X_train,y_train)
print("Best parameters: ", grid_search.best_params_)
print("Best Negative MSE score: ", grid_search.best_score_)

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
model_lr = LinearRegression()
accuracy_scores = cross_val_score(model_lr, X_train, y_train, cv = 10, scoring = 'neg_mean_squared_error')
mse_scores = -accuracy_scores
print("MSE Scores: ", mse_scores)

# Final Model
from sklearn.linear_model import LinearRegression
final = LinearRegression(fit_intercept = True, copy_X = True)
final.fit(X_train,y_train)

# Store 
import pickle
filename = 'LR_model.pkl'
with open(filename,'wb') as file:
    pickle.dump(final,file) 

# Pickle file is created and object is convert into binary in one file.
import pickle
import numpy as np
filename = 'LR_model.pkl'
with open(filename,'rb') as file:
    loaded_model = pickle.load(file)
height_input = 6
height_input_2d = np.array(height_input).reshape(1,-1)
predicted_weight = loaded_model.predict(height_input_2d)
print("Predicted weight: ",predicted_weight[0,0])    


