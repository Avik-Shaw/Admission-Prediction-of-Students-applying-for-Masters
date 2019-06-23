#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 13:09:22 2019

@author: avik
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn import model_selection
from sklearn import feature_selection
from sklearn import preprocessing
from sklearn import metrics
from sklearn import tree
from sklearn import neighbors
from sklearn import ensemble
from sklearn import naive_bayes

np.set_printoptions(suppress=True)
pd.set_option("display.max_columns",2500)
pd.set_option("display.max_rows",2500)

####################---pre-processing---#####################
org_dset=pd.read_csv("/home/avik_mint/MLproject/admission_prediction.csv")
##org_dset=pd.read_csv("E:/AVIK_glsn/MLproject/admission_prediction.csv")

org_dset.info()

org_dset["Chance of Admit"]# KeyError: 'Chance of Admit'
## this column is not recognized due to extra space at end
org_dset.columns#Index(['Serial No.', 'GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research', 'Chance of Admit '], dtype='object')
org_dset.columns=org_dset.columns.str.strip()
#Index(['Serial No.', 'GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research', 'Chance of Admit'], dtype='object')

org_dset.head()
## Serial No. has no use hence can be dropped

prc_dset=org_dset.drop("Serial No.",axis=1)

prc_dset.describe()
## count=500 for all columns means no missing values and other attributes showing no anomaly as the values are as per criteria
## Dataset is clean with all numeric values

####################---EDA---################################

sns.pairplot(data=prc_dset)
## GRE,CGPA & TOEFL scores are showing linearity with Chance of Admit
## CGPA v GRE & CGPA v TOEFL showing more GRE/TOEFL score more is the CGPA
## GRE v TOEFL showing students with high GRE score also have high TOEFL score
## students with high CGPA have high SOP & LOR

sns.countplot(x="University Rating",data=prc_dset)
## more students under Universities with rating 3

sns.countplot(x="Research",data=prc_dset)
## more students have done Research
sns.heatmap(prc_dset.corr(),annot=True)
## Research has lesser correlation with Chance of Admit
## Research is not much important for Chance of admission in this dataset
## So it can be dropped
## CGPA, GRE & TOEFL Score are highly correlated to each other
## so only one of them can be taken
## Again CGPA has highest correlation with Chance of Admiision(Target Variable)
## So CGPA should be chosen among the three for prediction model

## checking whether GRE,TOEFL,CGPA are normalized
sns.distplot(prc_dset["GRE Score"])
sns.distplot(prc_dset["TOEFL Score"])
sns.distplot(prc_dset["CGPA"])
## all are more or less normalized

plt.scatter(prc_dset["University Rating"],prc_dset["CGPA"])
plt.title("CGPA Scores with University Ratings")
plt.xlabel("University Rating")
plt.ylabel("CGPA")
## more CGPA earned by students of better University

df = prc_dset[prc_dset["Chance of Admit"]>=0.7]["University Rating"].value_counts()
df.plot(kind="bar")
plt.xlabel("University Rating")
plt.ylabel("No. of Students")
## so students from good univesities have more chance of admission

##############---Feature Scaling---###########################
X=prc_dset[["CGPA","University Rating","SOP","LOR"]]
y=prc_dset["Chance of Admit"]

Xtrain,Xtest,ytrain,ytest=model_selection.train_test_split(X,y,test_size=.2,random_state=42)


#######################---Regression model---####################

# cross-validation
lm=linear_model.LinearRegression()
scores_lm=model_selection.cross_val_score(lm,Xtrain,ytrain,scoring="r2",cv=5)
scores_lm.mean()#0.7720889092539394

tr=tree.DecisionTreeRegressor()
scores_tr=model_selection.cross_val_score(tr,Xtrain,ytrain,scoring="r2",cv=5)
scores_tr.mean()#0.554769358988724

knn=neighbors.KNeighborsRegressor()
scores_knn=model_selection.cross_val_score(knn,Xtrain,ytrain,scoring="r2",cv=5)
scores_knn.mean()#0.6793977255251684

rnf=ensemble.RandomForestRegressor()
scores_rnf=model_selection.cross_val_score(rnf,Xtrain,ytrain,scoring="r2",cv=5)
scores_rnf.mean()#0.7040949566141077


'''
scaling doesn't affect the result much
stdscalar=preprocessing.StandardScaler()
X_std=stdscalar.fit_transform(X)
X=pd.DataFrame(X_std,columns=X.columns)
Xtrain,Xtest,ytrain,ytest=model_selection.train_test_split(X,y,test_size=.2,random_state=42)
#######################---Regression model---####################

# cross-validation
lm=linear_model.LinearRegression()
scores_lm=model_selection.cross_val_score(lm,Xtrain,ytrain,scoring="r2",cv=5)
scores_lm.mean()#0.7720889092539392

tr=tree.DecisionTreeRegressor()
scores_tr=model_selection.cross_val_score(tr,Xtrain,ytrain,scoring="r2",cv=5)
scores_tr.mean()#0.5639985796245208


knn=neighbors.KNeighborsRegressor()
scores_knn=model_selection.cross_val_score(knn,Xtrain,ytrain,scoring="r2",cv=5)
scores_knn.mean()#0.7005613229223941

rnf=ensemble.RandomForestRegressor()
scores_rnf=model_selection.cross_val_score(rnf,Xtrain,ytrain,scoring="r2",cv=5)
scores_rnf.mean()#0.70738507562276
'''

plt.bar(["LinearRegression","DecisionTree","RandomForest","K-NN"],np.array([scores_lm.mean(),scores_tr.mean(),scores_rnf.mean(),scores_knn.mean()]))
plt.title("Performance Comparision of Regressor Models")
plt.xlabel("Regressor")
plt.ylabel("R2 Score")

## Linear Regression is the best model for this dataset

lm.fit(Xtrain,ytrain)
ypred=lm.predict(Xtest)

print("RMSE:",np.sqrt(np.mean((ypred-ytest)**2)))#0.05885931251689215
print("Adjusted R2 method:",lm.score(Xtest,ytest))#0.8305907740850281

#################---Classification---###########################

## If Chance of Admit>=0.7 --> 1(student will get admitted)
## If Chance of Admit<0.7 --> 0(student will not get admitted)

y=np.array([1 if coa>=0.7 else 0 for coa in y])

cl_dset=X
cl_dset["Chance of Admit"]=pd.Series(y)
sns.pairplot(data=cl_dset,hue="Chance of Admit")
## CGPA v Chance of Admit shows students with more CGPA are selected
## while those with less have less chance of selection

Xtrain,Xtest,ytrain,ytest=model_selection.train_test_split(X,y,test_size=.2,random_state=42)

lm=linear_model.LogisticRegression()
scores_lm=model_selection.cross_val_score(lm,Xtrain,ytrain,scoring="f1",cv=5)
scores_lm.mean()#0.8178502978502978

tr=tree.DecisionTreeClassifier()
scores_tr=model_selection.cross_val_score(tr,Xtrain,ytrain,scoring="f1",cv=5)
scores_tr.mean()#0.8146684559962256

knn=neighbors.KNeighborsClassifier()
scores_knn=model_selection.cross_val_score(knn,Xtrain,ytrain,scoring="f1",cv=5)
scores_knn.mean()#0.8554021372869934

rnf=ensemble.RandomForestClassifier()
scores_rnf=model_selection.cross_val_score(rnf,Xtrain,ytrain,scoring="f1",cv=5)
scores_rnf.mean()#0.84124271248795

gnb=naive_bayes.GaussianNB()
scores_gnb=model_selection.cross_val_score(gnb,Xtrain,ytrain,scoring="f1",cv=5)
scores_gnb.mean()#0.8477065844025319
## classification model scores are good but lets try scaling for more accuracy
## now scaling the values -- Standard Scalar
stdscalar=preprocessing.StandardScaler()
X_std=stdscalar.fit_transform(X)
X=pd.DataFrame(X_std,columns=X.columns)

Xtrain,Xtest,ytrain,ytest=model_selection.train_test_split(X,y,test_size=.2,random_state=42)

lm=linear_model.LogisticRegression()
scores_lm=model_selection.cross_val_score(lm,Xtrain,ytrain,scoring="f1",cv=5)
scores_lm.mean()#0.8672141017661265

tr=tree.DecisionTreeClassifier()
scores_tr=model_selection.cross_val_score(tr,Xtrain,ytrain,scoring="f1",cv=5)
scores_tr.mean()#0.8053145144301664

knn=neighbors.KNeighborsClassifier()
scores_knn=model_selection.cross_val_score(knn,Xtrain,ytrain,scoring="f1",cv=5)
scores_knn.mean()#0.8517196495619525

rnf=ensemble.RandomForestClassifier()
scores_rnf=model_selection.cross_val_score(rnf,Xtrain,ytrain,scoring="f1",cv=5)
scores_rnf.mean()#0.8181181782139394

gnb=naive_bayes.GaussianNB()
scores_gnb=model_selection.cross_val_score(gnb,Xtrain,ytrain,scoring="f1",cv=5)
scores_gnb.mean()#0.8477065844025319

## Robust Scalar

X=prc_dset[["CGPA","University Rating","SOP","LOR"]]

rbscalar=preprocessing.RobustScaler()
X_rb=rbscalar.fit_transform(X)
X=pd.DataFrame(X_rb,columns=X.columns)

Xtrain,Xtest,ytrain,ytest=model_selection.train_test_split(X,y,test_size=.2,random_state=42)

lm=linear_model.LogisticRegression()
scores_lm_rb=model_selection.cross_val_score(lm,Xtrain,ytrain,scoring="f1",cv=5)
scores_lm_rb.mean()#0.8672141017661265

tr=tree.DecisionTreeClassifier()
scores_tr_rb=model_selection.cross_val_score(tr,Xtrain,ytrain,scoring="f1",cv=5)
scores_tr_rb.mean()#0.8141116037912777

knn=neighbors.KNeighborsClassifier()
scores_knn_rb=model_selection.cross_val_score(knn,Xtrain,ytrain,scoring="f1",cv=5)
scores_knn_rb.mean()#0.8518016853787428

rnf=ensemble.RandomForestClassifier()
scores_rnf_rb=model_selection.cross_val_score(rnf,Xtrain,ytrain,scoring="f1",cv=5)
scores_rnf_rb.mean()#0.8298614422541049

gnb=naive_bayes.GaussianNB()
scores_gnb_rb=model_selection.cross_val_score(gnb,Xtrain,ytrain,scoring="f1",cv=5)
scores_gnb_rb.mean()#0.8477065844025319

## Min-Max Scalar

X=prc_dset[["CGPA","University Rating","SOP","LOR"]]

mmscalar=preprocessing.MinMaxScaler()
X_mm=mmscalar.fit_transform(X)
X=pd.DataFrame(X_mm,columns=X.columns)

Xtrain,Xtest,ytrain,ytest=model_selection.train_test_split(X,y,test_size=.2,random_state=42)

lm=linear_model.LogisticRegression()
scores_lm=model_selection.cross_val_score(lm,Xtrain,ytrain,scoring="f1",cv=5)
scores_lm.mean()#0.8592826977565995

tr=tree.DecisionTreeClassifier()
scores_tr=model_selection.cross_val_score(tr,Xtrain,ytrain,scoring="f1",cv=5)
scores_tr.mean()#0.8035915974519042

knn=neighbors.KNeighborsClassifier()
scores_knn=model_selection.cross_val_score(knn,Xtrain,ytrain,scoring="f1",cv=5)
scores_knn.mean()#0.8501569579288025

rnf=ensemble.RandomForestClassifier()
scores_rnf=model_selection.cross_val_score(rnf,Xtrain,ytrain,scoring="f1",cv=5)
scores_rnf.mean()#0.8459883040935672

gnb=naive_bayes.GaussianNB()
scores_gnb=model_selection.cross_val_score(gnb,Xtrain,ytrain,scoring="f1",cv=5)
scores_gnb.mean()#0.8477065844025319

## Robust & Standard Scaling both gives optimum results
## We have chosen Robust Scaling in this case

plt.bar(["LogisticRegression","DecisionTree","RandomForest","K-NN","GNB"],np.array([scores_lm_rb.mean(),scores_tr_rb.mean(),scores_rnf_rb.mean(),scores_knn_rb.mean(),scores_gnb_rb.mean()]))
plt.title("Performance Comparision of Classification Models")
plt.xlabel("Classifier models")
plt.ylabel("F1 Score")

## Logistic Regression is the best classifier in this case

X=prc_dset[["CGPA","University Rating","SOP","LOR"]]

rbscalar=preprocessing.RobustScaler()
X_rb=rbscalar.fit_transform(X)
X=pd.DataFrame(X_rb,columns=X.columns)
lm=linear_model.LogisticRegression()
lm.fit(Xtrain,ytrain)
ypred=lm.predict(Xtest)

confmat=metrics.confusion_matrix(ytest,ypred)

sns.heatmap(data=confmat,annot=True)

print("Accuracy Score:",metrics.accuracy_score(ytest,ypred))#Accuracy Score: 0.81
print("Precision Score:",metrics.precision_score(ytest,ypred))#Precision Score: 0.7727272727272727
print("Recall Score:",metrics.recall_score(ytest,ypred))#Recall Score: 0.9272727272727272
print("F1 Score:",metrics.f1_score(ytest,ypred))#F1 Score: 0.8429752066115703
print("AUC Score:",metrics.roc_auc_score(ytest,ypred))#AUC Score: 0.796969696969697
