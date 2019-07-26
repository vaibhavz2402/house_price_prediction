# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 19:15:17 2019

@author: Vaibhav
"""
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(train.info())
print(test.info())

n_train = train.shape[0]
n_test = test.shape[0]
y = train['SalePrice'].values

data = pd.concat((train, test)).reset_index(drop = True)
data.drop(['SalePrice'], axis=1, inplace = True)

#Data Visualisation
plt.figure()
sns.heatmap(train.corr(),cmap='coolwarm')
plt.show()

sns.pairplot(train, palette='rainbow')

sns.lmplot(x='YearBuilt', y='SalePrice', data=train)

sns.boxplot(x='GarageCars', y='SalePrice', data=train)

sns.barplot(x='GarageArea', y='SalePrice', data= train, estimator=np.mean)

sns.barplot(x='FullBath',y = 'SalePrice',data=train)

sns.lmplot(x='1stFlrSF',y='SalePrice',data=train)

#Feature Engineering
data = data[['LotArea','Street', 'Neighborhood','Condition1', 'Condition2','BldgType','HouseStyle','OverallCond', 'Heating','CentralAir','Electrical','1stFlrSF','2ndFlrSF','BsmtHalfBath','FullBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','GarageCars','GarageArea','PoolArea']]
data.info()

data['BsmtHalfBath'] = data['BsmtHalfBath'].fillna(data['BsmtHalfBath'].mean())
data['Electrical'] = data['Electrical'].fillna('SBrkr')
data['GarageCars'] = data['GarageCars'].fillna(data['GarageCars'].mean())
data['GarageArea'] = data['GarageArea'].fillna(data['GarageArea'].mean())

#Encoding categorical features
categorical_feature_mask = data.dtypes==object
categorical_cols = data.columns[categorical_feature_mask].tolist()

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
data[categorical_cols] = data[categorical_cols].apply(lambda col: labelencoder.fit_transform(col))

data.shape

train = data[:n_train]
test = data[n_train:]

#linear Regression
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size = 0.2, random_state = 0)

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.fit_transform(y_test)
 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

 