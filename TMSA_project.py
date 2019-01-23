# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 23:53:57 2018

@author: Vignesh
"""
import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from hyperopt import hp, tpe, fmin,STATUS_OK, Trials,space_eval
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold  
from sklearn.svm import SVC
from sklearn import neighbors



# Random forest bayesian hyperoptimization,crossvalidation and holdout data scoring
def randomForest(X_train,y_train,X_test,y_test,criterion,max_depth,max_features,n_estimators):
    ran= RandomForestClassifier(criterion= criterion, max_depth= max_depth, max_features= max_features, n_estimators= n_estimators,random_state=17027)
    ran.fit(X_train,y_train)
    return ran.score(X_test,y_test)

def rf_acc_model(params):
    clf = RandomForestClassifier(**params)
    shuffle = KFold(n_splits=10)
    score=cross_val_score(clf, X_train, Y_train,cv=shuffle)
    return score.mean()

def rf_f(params):
    best=0
    acc = rf_acc_model(params)
    if acc > best:
        best = acc
    print ('new best:', best, params)
    return {'loss': -acc, 'status': STATUS_OK}


def rfhyperoptimizationparameters(X_train,Y_train):
    
    rf_param_space = {
        'max_depth': hp.choice('max_depth', range(1,20)),
        'max_features': hp.choice('max_features', range(1,150)),
        'n_estimators': hp.choice('n_estimators', range(100,500)),
        'criterion': hp.choice('criterion', ["gini", "entropy"])}
    
    best = 0
    
    trials = Trials()
    best = fmin(rf_f, rf_param_space, algo=tpe.suggest, max_evals=1, trials=trials)
    print ('best:')
    rfbestHyper=space_eval(rf_param_space, best)
    print (rfbestHyper)
    return rfbestHyper

# Support vector machine bayesian  hyperoptimization,crossvalidation and holdout data scoring
def suvcm(X_train,y_train,X_test,y_test,C,kernel,gamma):
    print("Entered svm accuracy determination")
    s= SVC(C=C,kernel=kernel,gamma=gamma,random_state=17027)
    s.fit(X_train,y_train)
    return s.score(X_test,y_test)

def svm_acc_model(params):
    clf = SVC(**params)
    shuffle = KFold(n_splits=10)
    score=cross_val_score(clf, X_train, Y_train,cv=shuffle)
    return score.mean()

def svm_f(params):
    best=0
    acc = svm_acc_model(params)
    if acc > best:
        best = acc
    print ('new best:', best, params)
    return {'loss': -acc, 'status': STATUS_OK}


def svmhyperoptimizationparameters(X_train,Y_train):
    print("Entered svm")
    svm_param_space = {
    'C': hp.uniform('C', 0, 20),
    'kernel': hp.choice('kernel', ['linear', 'sigmoid', 'poly', 'rbf']),
    'gamma': hp.uniform('gamma', 0, 20)}
    
    best = 0
    
    trials = Trials()
    best = fmin(svm_f, svm_param_space, algo=tpe.suggest, max_evals=1, trials=trials)
    print ('best:')
    svmbestHyper=space_eval(svm_param_space, best)
    print (svmbestHyper)
    return svmbestHyper

# K-Nearest Neighbors bayesian  hyperoptimization,crossvalidation and holdout data scoring
def knnClass(X_train,y_train,X_test,y_test,n_neighbors):
    k= neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    k.fit(X_train,y_train)
    return k.score(X_test,y_test)

def knn_acc_model(params):
    clf = neighbors.KNeighborsClassifier(**params)
    shuffle = KFold(n_splits=10,random_state=17027)
    score=cross_val_score(clf, X_train, Y_train,cv=shuffle)
    return score.mean()

def knn_f(params):
    best=0
    acc = knn_acc_model(params)
    if acc > best:
        best = acc
    print ('new best:', best, params)
    return {'loss': -acc, 'status': STATUS_OK}


def knnhyperoptimizationparameters(X_train,Y_train):
    print("Entered knn")
    knn_param_space = {
    'n_neighbors': hp.choice('n_neighbors', range(1,50))}
    
    best = 0
    
    trials = Trials()
    best = fmin(knn_f, knn_param_space, algo=tpe.suggest, max_evals=1, trials=trials)
    print ('best:')
    knnbestHyper=space_eval(knn_param_space, best)
    print (knnbestHyper)
    return knnbestHyper

start=time.time()
# Data preparation
Data= pd.read_excel("D:/IIM/Term 5/Text Mining and Social Media Analytics/project/sentiment_analysis_clean.xlsx")
data=Data.copy()
data=data.dropna(subset=['Sentiment', 'SentimentText'])
A=data.index[data["SentimentText"] == 0 ].tolist()
data=data.drop(data.index[A])
data=data.dropna()
#vectorizing the text
tfidf = TfidfVectorizer(min_df=1,stop_words='english')
response = tfidf.fit_transform(data['SentimentText'])
print(len)
tfidfmatrix=response.toarray()
dx=tfidfmatrix
dy=data['Sentiment']
xtrain,xtest,ytrain,ytest=train_test_split(dx,dy,test_size = 0.25,random_state=17027)

X_train=xtrain
Y_train=ytrain
print("Total time taken for execution in seconds :",((time.time()-start)))

# Random forest Cross validation,hyperoptimization and accuracy
rfbest=rfhyperoptimizationparameters(X_train,Y_train)
rfAccuracy=randomForest(xtrain,ytrain,xtest,ytest,**rfbest)
print("Random forest Accuracy",rfAccuracy)

# Support vector machine Cross validation,hyperoptimization and accuracy
svmbest=svmhyperoptimizationparameters(X_train,Y_train)
svmAccuracy=suvcm(xtrain,ytrain,xtest,ytest,**svmbest)
print("SVM Accuracy",svmAccuracy)

# K-nearest neighbors Cross validation,hyperoptimization and accuracy
knnbest=knnhyperoptimizationparameters(X_train,Y_train)
knnAccuracy=knnClass(xtrain,ytrain,xtest,ytest,**knnbest)
print("KNN Accuracy",knnAccuracy)


# Final scores
print ("\n\n***************************\n\n")
print("Random forest Best hyper parameters\n",rfbest)
print("Support vector Best hyper parameters\n",svmbest)
print("KNN  Best hyper parameters\n",knnbest)
print("Random forest Accuracy=",rfAccuracy)
print("Support Vector Machine Accuracy=",svmAccuracy)
print("K-nearest neighbors Accuracy=",knnAccuracy)
print("Total time taken for execution in seconds :",((time.time()-start)))
print("Total time taken for execution in hr :",((time.time()-start)/3600))

print ("\n\n***************************\n\n\n\n")






