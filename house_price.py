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

