# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 12:21:04 2020

@author: bnola
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
import tensorflow as tf
from tensorflow import keras
import seaborn as sns

def NHL_player_data_retrieval(): # retrieve data from excel file and return non-goalie player stats 
    df = pd.read_excel("C:\\Users\\bnola\\Documents\\Coding practice\\2018-2019 NHL Stats+Salary.xlsx")
    df = pd.DataFrame(df)
    df = df.dropna(axis=0, how='any')
    df = df.mask(df['POS']=='G')
    df = df.dropna(axis=0, how='any')
    X = df.drop(['TOI', 'PLAYER', 'TEAM', 'POS', 'HANDED', 'CAP HIT', 'SALARY'], axis=1)
    y = df['SALARY'] # AAV or CAP Hit also possible options
    y = y/1000000.00 #labeled data for supervised learning (continuous variable)
    return X, y

def descriptive_plot(X, y):
    fig = plt.figure(figsize=(10,7))
    fig.add_subplot(2,1,1)
    ax1 = sns.histplot(data=y)
    ax1.set(xlabel='Salary (millions)') #distribution of NHL player salaries
    fig.add_subplot(2,1,2)
    ax2 = sns.boxplot(data=y, orient='h') #boxplot of salaries
    ax2.set(xlabel='Salary (millions)')
    plt.tight_layout()
    fig = plt.figure(figsize=(18,5));
    fig.add_subplot(2,2,1)
    sns.scatterplot(x=y, y= X['P']) 
    fig.add_subplot(2,2,2)
    sns.scatterplot(x=y,y= X['G'])
    fig.add_subplot(2,2,3)
    sns.scatterplot(x=y,y= X['Sh'])
    fig.add_subplot(2,2,4)
    sns.scatterplot(x=y,y= X['AGE'])
    fig.tight_layout()
    return 

def split_preprocess(X, y):
    from sklearn.model_selection import train_test_split
    from sklearn import preprocessing
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2)
    s_scaler = preprocessing.StandardScaler() #standardizing features by removing the mean and scaling to unit variance
    X_train = s_scaler.fit_transform(X_train)
    X_test = s_scaler.transform(X_test)
    return X_train, X_test, Y_train, Y_test

def LRPredict(X, X_train, X_test, Y_train, Y_test):
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression() 
    regressor.fit(X_train, Y_train)
    y_pred_reg = regressor.predict(X_test)
    coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
    print(regressor.intercept_)
    print(regressor.coef_)
    print(regressor.score(X_test,Y_test))
    return regressor, y_pred_reg

def LR_evaluation(X_train, X_test, Y_train, Y_test, y_pred_reg): # evaluate the performance of the algorithm (MAE - MSE - RMSE)
    from sklearn import metrics
    print('MAE:', metrics.mean_absolute_error(Y_test, y_pred_reg))  
    print('MSE:', metrics.mean_squared_error(Y_test, y_pred_reg))  
    print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred_reg)))
    print('VarScore:',metrics.explained_variance_score(Y_test,y_pred_reg))
    df = pd.DataFrame({'Actual': Y_test, 'Predicted': y_pred_reg})
    plt.figure(figsize=(10,5));
    plt.scatter(df['Actual'], df['Predicted']);
    plt.plot(df['Actual'], df['Actual'],'r');
    plt.xlabel('Actual Salary')
    plt.ylabel('Predicted Salary')
    resids = (Y_test - y_pred_reg)
    sns.displot(resids);
    return

def nn_model(X_train, X_test, Y_train, Y_test, nodes):
    model = keras.Sequential([keras.layers.Dense(nodes, activation='relu'), 
                          keras.layers.Dense(nodes, activation='relu'),
                          keras.layers.Dense(nodes, activation='relu'),
                          keras.layers.Dense(1)]) #4-layer Neural net, with n number of nodes
    model.compile(loss='mae', #mean absolute error loss function
                optimizer='Adam') #RMSprop with momentum optimization function 
    history = model.fit(x=X_train, y=Y_train, epochs=200, verbose=1, validation_data=(X_test, Y_test))
    y_pred = model.predict(X_test)
    return history, y_pred

def nn_model_evaluation(Y_test, y_pred, history):
    fig = plt.figure(figsize=(10,5))
    plt.scatter(Y_test,y_pred)
    plt.plot(Y_test,Y_test,'r')
    plt.xlabel('Actual Salary (millions)')
    plt.ylabel('Predicted Salary (millions)')
    from sklearn import metrics
    print('MAE:', metrics.mean_absolute_error(Y_test, y_pred))  
    print('MSE:', metrics.mean_squared_error(Y_test, y_pred))  
    print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))
    print('VarScore:',metrics.explained_variance_score(Y_test,y_pred))
    loss_df = pd.DataFrame(history.history)
    loss_df.plot(figsize=(10,6))
    return
