#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 09:30:58 2018

@author: vikramreddy
"""

#######################
#
#
# here total count is the sum of registered and casual users
# we have to predict casual and registered users and sum them and
#compare to count(find RMSE on total count)
#
#
#######################


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#loading data
train=pd.read_csv('day.csv')
dat_main=train.copy()
#encoding of factor variables 
rule0={1:"Spring", 2:"Summer",3: "Fall",4: "Winter"}
rule1={ 1:"clear",2:"mist",3:"light snow"}
rule2={0:2011,1:2012}
rule3={1:'jan',2:'feb',3:'march',4:'apr',5:'may',6:'jun',7:'jul',8:'aug',9:'sep',
                                      10:'oct',11:'nov',12:'dec'}
rule4={0:'No',1:'Yes'}
rule5={0:'sun',1:'mom',2:'tue',3:'wed',4:'thu',5:'fri',6:'sat'}
train['season']=train['season'].replace(rule0)
train['weathersit']=train['weathersit'].replace(rule1)
train['yr']=train['yr'].replace(rule2)
train['mnth']=train['mnth'].replace(rule3)
train['holiday']=train['holiday'].replace(rule4)
train['workingday']=train['workingday'].replace(rule4)
train['weekday']=train['weekday'].replace(rule5)
dat=train.copy()

train.isnull().any().any()
# There is no missing data in the data set

#########################
#
#Exploratory data analysis
#
#########################
#histogram of categorical data

sns.countplot(dat['season'], color='orange')
sns.countplot(dat['weathersit'], color='red')
sns.countplot(dat['yr'], color='yellow')
sns.countplot(dat['mnth'], color='blue')
sns.countplot(dat['holiday'], color='pink')
sns.countplot(dat['workingday'], color='green')
sns.countplot(dat['weekday'], color='black')

#checking for outliers in the data(continuous variables)
sns.boxplot(dat['cnt'])
sns.boxplot(dat['temp'])
sns.boxplot(dat['registered'])
sns.boxplot(dat['casual'])
sns.boxplot(dat['atemp'])
sns.boxplot(dat['hum'])
sns.boxplot(dat['windspeed'])

#distribution of categorical data vs count
sns.boxplot(x='season', y='cnt', data=dat)
sns.boxplot(x='weathersit', y='cnt', data=dat)
sns.boxplot(x='yr', y='cnt', data=dat)
sns.boxplot(x='mnth', y='cnt', data=dat)
sns.boxplot(x='holiday', y='cnt', data=dat)
sns.boxplot(x='weekday', y='cnt', data=dat)
sns.boxplot(x='workingday', y='cnt', data=dat)
##################################
#
#
#
# plotting registered users and casual users across 
#different feature
#
#
#
#################################

sns.boxplot(x='season', y='registered', data=dat)
sns.boxplot(x='weathersit', y='registered', data=dat)
sns.boxplot(x='yr', y='registered', data=dat)
sns.boxplot(x='mnth', y='registered', data=dat)
sns.boxplot(x='holiday', y='registered', data=dat)
sns.boxplot(x='weekday', y='registered', data=dat)
sns.boxplot(x='workingday', y='registered', data=dat)



sns.boxplot(x='season', y='casual', data=dat)
sns.boxplot(x='weathersit', y='casual', data=dat)
sns.boxplot(x='yr', y='casual', data=dat)
sns.boxplot(x='mnth', y='casual', data=dat)
sns.boxplot(x='holiday', y='casual', data=dat)
sns.boxplot(x='weekday', y='casual', data=dat)
sns.boxplot(x='workingday', y='casual', data=dat)

###########################
#
#
#Feature engineering
#
#
##########################

#correlation plot
colormap = plt.cm.RdBu
plt.figure(figsize=(15,15))
plt.title('Pearson Correlation of Features', y=1.0, size=10)
sns.heatmap(train[['cnt','temp',
 'atemp',
 'hum',
 'windspeed',
 'casual',
 'registered',]].corr(),linewidths=0.2,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
#"atemp" is variable is not taken into since "atemp" and "temp" has 
#got strong correlation with each other. 
#During model building any one of the variable has to be dropped since 
#they will exhibit multicollinearity in the data.

train=train.drop(['atemp'],axis=1)

################
#
# creating new feature
#Creating bins for casual casual variable based on its relation with
# month column
#
#
###############
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn import tree
DT_cas_mnth=DecisionTreeRegressor(max_depth=2)
dat_main['mnth']=dat_main['mnth'].astype('category')
DT_cas_mnth.fit(dat_main['casual'].values.reshape(-1,1),dat_main['mnth'])
feat_cas=list(train.columns[13:14])
tar_mnth=list(train.columns[4:5])
#exporting the graph
############
#  stepd to veiw this tree
#1.open the .dot file in text editor
#2.copy all the code
#3.go to the link:http://webgraphviz.com/
#4.paste the code and run
#5.you will geta graph
#
# I have attached the grapg image in report
##############

tree.export_graphviz(DT_cas_mnth,out_file='tr.dot',feature_names = feat_cas,class_names=tar_mnth)

#on the basis of above graph i have create a new varible  
cas_mnth=pd.Series([])

for i in range(731):
    if(train.iloc[i,13] <= 142):
        cas_mnth[i]=1
    else:
        cas_mnth[i]=2

for i in range(731):
    if(cas_mnth[i]==1):
        if(train.iloc[i,13] <= 12):
            cas_mnth[i]=3
        else:
            cas_mnth[i]=4
    else:
        ''


for i in range(731):
    if(cas_mnth[i]==2):
        if(train.iloc[i,13] <= 253.5):
            cas_mnth[i]=5
        else:
            cas_mnth[i]=6
    else:
        ''
 

################
#
# creating new feature
#Creating bins for registered variable based on its relation with
# month column
#
#
###############                       
DT_reg_mnth=DecisionTreeRegressor(max_depth=2)     
DT_reg_mnth.fit(dat['registered'].values.reshape(-1,1),dat['mnth'])
target_mnth=list(dat.columns[4:5])
feat_reg=list(dat.columns[14:15])



tree.export_graphviz(DT_reg_mnth,out_file='tr1.dot',feature_names = feat_reg,class_names=target_mnth)
############
#  stepd to veiw this tree
#1.open the .dot file in text editor
#2.copy all the code
#3.go to the link:http://webgraphviz.com/
#4.paste the code and run
#5.you will geta graph
#
# I have attached the grapg image in report
###############
reg_mnth=pd.Series([])

for i in range(731):
    if(train.iloc[i,14] <= 2111.5):
        reg_mnth[i]=1
    else:
        reg_mnth[i]=2

for i in range(731):
    if(reg_mnth[i]==1):
        if(train.iloc[i,14] <= 575.0):
            reg_mnth[i]=3
        else:
            reg_mnth[i]=4
    else:
        ''


for i in range(731):
    if(reg_mnth[i]==2):
        if(train.iloc[i,14] <= 6457.5):
            reg_mnth[i]=5
        else:
            reg_mnth[i]=6
    else:
        ''
#merging newly created varibles to the main data

train=train.merge(cas_mnth.to_frame(), left_index=True, right_index=True)
train=train.merge(reg_mnth.to_frame(), left_index=True, right_index=True)

#replacing the valus that lie greather than 0.95 quantile with the value
#that lie in o.95 quantile

train.casual.quantile([0.95]) #value is 2355
train.casual=train.casual.mask(train.casual >2355,2355)
#as the casual variable is changed
# so we have to update total count variables as it is sum of casual and registered users
train.cnt=train.casual+train.registered
  
###########################
#
#
#
# Feature engineering
#
#
###########################
#conveting this variables in to their respective type
train['mnth']=train['mnth'].astype('category')
train['season']=train['season'].astype('category')
train['weekday']=train['weekday'].astype('category')
train['weathersit']=train['weathersit'].astype('category')
train['workingday']=train['workingday'].astype('category')
train['workingday']=train['workingday'].astype('category')
train['holiday']=train['holiday'].astype('category')
#convertinf newly creaed variables to category type
train['0_y']=train['0_y'].astype('category')
train['0_x']=train['0_x'].astype('category')



################
#
#One hot encoding of factor varibles
#
#
###############

mnth_dummy=pd.get_dummies(train['mnth'])
season_dummy=pd.get_dummies(train['season'])
weekday_dummy=pd.get_dummies(train['weekday'])
weather_dummy=pd.get_dummies(train['weathersit'])
working_dummy=pd.get_dummies(train['workingday'])
holiday_dummy=pd.get_dummies(train['holiday'])
yr_dummy=pd.get_dummies(train['yr'])

holiday_dummy.columns=['no_hol','yes_hol']

#combining one hot encoded column to the main data
train=train.join([mnth_dummy,season_dummy,weekday_dummy,weather_dummy,working_dummy,holiday_dummy,yr_dummy])
#removing that factor columns for which we have done one hot encoding
train=train.drop(['mnth','season','weekday','weathersit','workingday','instant',
 'dteday','holiday','yr'],axis=1)
    
    #importing models
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression,ElasticNet
from sklearn.ensemble import  RandomForestRegressor
from sklearn.metrics import mean_squared_error

from sklearn.utils import shuffle
#shuffling data
train = shuffle(train)
###################
#
#Feature selection using select K best
#
#
###################
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import scale
#data for predicting  casual count
train_cas=train.drop(['casual','registered','cnt'],axis=1)
#target (casual count)
test_cas=train['casual']
##data for predicting  registered count

train_reg=train.drop(['casual','registered','cnt'],axis=1)
##target  (registered count)

test_reg=train['registered']
#selecting top 30 features
train
X_new_cas = SelectKBest(f_regression, k=30).fit_transform(train_cas,test_cas)
X_new_reg = SelectKBest(f_regression, k=30).fit_transform(train_reg,test_reg)

##########################
#
# spliting manually because we are adding predictions(casual+registerd) and compare to toal count which does not belong 
#two components(random state will shuffle data)
###################

#################
#
# casual users
#
###################



x_train_cas=X_new_cas[:550:1]
x_test_cas=X_new_cas[551::1]
y_train_cas=test_cas.iloc[:550,]
y_test_cas=test_cas.iloc[551:,]

x_train_cas=train_cas.iloc[:550,:]
x_test_cas=train_cas.iloc[551:,:]
y_train_cas=test_cas.iloc[:550,]
y_test_cas=test_cas.iloc[551:,]

#################
#
#registerd users
#
###################

x_train_reg=X_new_reg[:550:1]
x_test_reg=X_new_reg[551::1]
#x_train_reg=train_reg.iloc[:550,:]
#x_test_reg=train_reg.iloc[551:,:]
y_train_reg=test_reg.iloc[:550,]
y_test_reg=test_reg.iloc[551:,]

test_count=train.iloc[551:,5]

#####################
#
#
#creating  function model for casual and registerd users and summing their predictions
#and comparing with total count of test data
#
#
#
#####################



def model_results(model):
    
    ###########casual users
    
   #fitting the data
    model.fit(x_train_cas,y_train_cas)
    
    #test predictions
    test_predictions_cas=model.predict(x_test_cas)
    #RMSE of test data
    
    RMSE_cas=np.sqrt(mean_squared_error(y_test_cas, test_predictions_cas))
    
    print("test-RMSE of casual user count ")
    print(RMSE_cas)
    print('coefficient of determination R^2 of the prediction')
    #model score on test data

    print(model.score(x_test_cas, y_test_cas))
    
    
    ######registered users
    
    #fitting the raw data(with outliers tothe model)
    model.fit(x_train_reg,y_train_reg)
    
    #test predictions
    test_predictions_reg=model.predict(x_test_reg)
    #RMSE on test data
    RMSE_reg=np.sqrt(mean_squared_error(y_test_reg, test_predictions_reg))    
    print("test-RMSE of registered user count ")
    print(RMSE_reg)
    print('coefficient of determination R^2 of the prediction')
    #model score on test data
    print(model.score(x_test_reg, y_test_reg))
    
     #summing casual and registered predictions to get total count predictions
    
    count_predictions=test_predictions_reg+test_predictions_cas
    
    #finding RMSE on total count(by summing up predictions)
    RMSE_count=np.sqrt(mean_squared_error(count_predictions, test_count))
    print('RMSE of total count(registered+casual)')
    print(RMSE_count)
   

    return ""
 
    
#ridge regression
Ridge_model=Ridge()#***
model_results(Ridge_model)


#Linear regression
    
lin_reg_model=LinearRegression()#****
model_results(lin_reg_model)


#Lasso
lasso_model=Lasso()
model_results(lasso_model)

#elatic net
ela_net=ElasticNet()
model_results(ela_net)

#####################
#
#Regression trees
#
#
###################

#random forest
rf_model=RandomForestRegressor()
model_results(rf_model)

#Decision tree regressor
DT_model=DecisionTreeRegressor()
model_results(DT_model)

   






#################
#
#
# PCA on registered and casual
#
#
#################
data_cas=train.drop(['registered','cnt','0_y','0_x'],axis=1)
data_reg=train.drop(['casual','cnt','0_y','0_x'],axis=1)
X=data_cas.values
#X=scale(X)
X1=data_reg.values
#X1=scale(X1)
#total count(casual +registered) of test data

test_count=train.iloc[551:,5]


#target variable
target_pca_cas=train['casual']
target_pca_reg=train['registered']


#passing the total number of components to the PCA   
from sklearn.decomposition import PCA
 
pca_cas = PCA(n_components=36)
pca_reg=PCA(n_components=36)

#fitting the values to PCA
pca_cas.fit(scale(X))
pca_reg.fit(scale(X1))

#pca_digits=PCA(0.99)
#X1_cas = pca_digits.fit_transform(X)
#X1_reg=pca_digits.fit_transform(X1)

#The amount of variance that each PC explained
var_cas= pca_cas.explained_variance_ratio_
var_reg= pca_reg.explained_variance_ratio_

    
#Cumulative Variance 
var1_cas=np.cumsum(np.round(pca_cas.explained_variance_ratio_, decimals=4)*100)
var1_reg=np.cumsum(np.round(pca_reg.explained_variance_ratio_, decimals=4)*100)


#graph of the variance
    
plt.plot(var1_cas)
plt.plot(var1_reg)



#############################
## from the above plot 
#The plot above shows that all components explains around 99% variance in the data set. 
#
#############################  

 
#Looking at above plot I'm taking 40 variables
pca_cas = PCA(n_components=25)
pca_reg = PCA(n_components=25)


#now fitting the selected components to the data
pca_cas.fit(X)
pca_cas.fit(X1)


#PCA selected features
X1_cas=pca_cas.fit_transform(X)
X1_reg=pca_reg.fit_transform(X1)

#splitting train and test data

x_train_pca_cas=X1_cas[:550:1]
x_test_pca_cas=X1_cas[551::1]
y_train_pca_cas=target_pca_cas.iloc[:550,]
y_test_pca_cas=target_pca_cas.iloc[551:,]

x_train_pca_reg=X1_reg[:550:1]
x_test_pca_reg=X1_reg[551::1]
y_train_pca_reg=target_pca_reg.iloc[:550,]
y_test_pca_reg=target_pca_reg.iloc[551:,]




#################################
#
#
#creating a function that displays PCA results of their models
#
#
################################

def pca_model_results(model):
    
    
    
    #fitting training data to the model
    model.fit(x_train_pca_cas,y_train_pca_cas)
   
    

    
    
    #test predictions
    test_pred_pca_cas=model.predict(x_test_pca_cas)
    
    #RMSE of test predictions and test data
    RMSE=np.sqrt(mean_squared_error(y_test_pca_cas, test_pred_pca_cas))
    print("test-RMSE PCA model ")
    print(RMSE)
    
    
    # Returns the coefficient of determination R^2 of the prediction.
    print('coefficient of determination R^2 of the prediction')
    print(model.score(x_test_pca_cas, y_test_pca_cas))
    
    
    
    
    #fitting training data to the model
    model.fit(x_train_pca_reg,y_train_pca_reg)
   
    

    
    
    #test predictions
    test_pred_pca_reg=model.predict(x_test_pca_reg)
    
    #RMSE of test predictions and test data
    RMSE=np.sqrt(mean_squared_error(y_test_pca_reg, test_pred_pca_reg))
    print("test-RMSE PCA model ")
    print(RMSE)
    
    
    # Returns the coefficient of determination R^2 of the prediction.
    print('coefficient of determination R^2 of the prediction')
    print(model.score(x_test_pca_reg, y_test_pca_reg))
    
    
    #summing casual and registered predictions to get total count predictions
    
    count_predictions=test_pred_pca_reg+test_pred_pca_cas
    RMSE_count=np.sqrt(mean_squared_error(count_predictions, test_count)) 
    print('RMSE of total count(registered+casual)')
    print(RMSE_count)
    return ""



#####################
#
# Regularisation methods
#
#####################
    

#ridge regression
Ridge_model=Ridge()#***
pca_model_results(Ridge_model)


#Linear regression
    
lin_reg_model=LinearRegression()#****
pca_model_results(lin_reg_model)


#Lasso
lasso_model=Lasso()
pca_model_results(lasso_model)

#elatic net
ela_net=ElasticNet()
pca_model_results(ela_net)

#####################
#
#Regression trees
#
#
###################

#random forest
rf_model=RandomForestRegressor()
pca_model_results(rf_model)

#Decision tree regressor
DT_model=DecisionTreeRegressor()
pca_model_results(DT_model)


