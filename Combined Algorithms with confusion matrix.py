#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss
from collections import Counter
import math
from scipy import stats
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import accuracy_score
from sklearn.ensemble import StackingClassifier

def preprocess_dataset(dataset_path):
    # Your code starts here (Multiple lines)
    data = pd.read_csv(dataset_path)
    data = data.fillna(0)
    data = data.drop('user_id', axis=1)
    numcols = data.select_dtypes(exclude='object')
    catcols = data.select_dtypes(include='object')
    #player_df = [numcols+catcols]
    traindf = pd.concat([numcols,pd.get_dummies(catcols)],axis=1)
    features = traindf.columns
    traindf = pd.DataFrame(traindf,columns=features)
    y = traindf['great_customer_class']
    X = traindf.copy()
    del X['great_customer_class']
    num_feats=30
    feature_name = list(X.columns)
    # Your code ends here
    return X, y, num_feats

def cor_selector(X, y,num_feats):
    # Your code goes here (Multiple lines)
    coor_list=[]
    for i in X.columns.tolist():
        cor=np.corrcoef(X[i],y)[0,1]
        coor_list.append(cor)
    cor_list=[0 if np.isnan(i) else i for i in coor_list]
    cor_features=X.iloc[:,np.argsort(np.abs(coor_list))[-num_feats:]].columns.tolist()
    cor_support=[True if i in cor_features else False for i in X.columns.tolist()]    
    # Your code ends here
    return cor_support, cor_features

def chi_squared_selector(X, y, num_feats):
    X_norm = MinMaxScaler().fit_transform(X)
    chi_selector = SelectKBest(chi2, k=num_feats)
    chi_selector.fit(X_norm, y)
    chi_support = chi_selector.get_support()
    chi_feature = X.loc[:,chi_support].columns.tolist()
    return chi_support, chi_feature

def rfe_selector(X, y, num_feats):
    X_norm = MinMaxScaler().fit_transform(X)
    rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=num_feats, step=10, verbose=5)
    rfe_selector.fit(X_norm, y)
    rfe_support = rfe_selector.get_support()
    rfe_feature = X.loc[:,rfe_support].columns.tolist()
    return rfe_support, rfe_feature

def embedded_log_reg_selector(X, y, num_feats):
#     X_norm = MinMaxScaler().fit_transform(X)
    embedded_lr_selector = SelectFromModel(LogisticRegression(penalty="l2"), max_features=num_feats)
    embedded_lr_selector.fit(X, y)
    embedded_lr_support = embedded_lr_selector.get_support()
    embedded_lr_feature = X.loc[:,embedded_lr_support].columns.tolist()
    return embedded_lr_support, embedded_lr_feature

def embedded_rf_selector(X, y, num_feats):
    embedded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=num_feats)
    embedded_rf_selector.fit(X, y)
    embedded_rf_support = embedded_rf_selector.get_support()
    embedded_rf_feature = X.loc[:,embedded_rf_support].columns.tolist()
    return embedded_rf_support, embedded_rf_feature

def embedded_lgbm_selector(X, y, num_feats):
    lgbc=LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,
            reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)

    embedded_lgbm_selector = SelectFromModel(lgbc, max_features=num_feats)
    embedded_lgbm_selector.fit(X, y)
    embedded_lgbm_support = embedded_lgbm_selector.get_support()
    embedded_lgbm_feature = X.loc[:,embedded_lgbm_support].columns.tolist()
    return embedded_lgbm_support, embedded_lgbm_feature

def autoFeatureSelector(dataset_path, methods=[]):
    # Parameters
    # data - dataset to be analyzed (csv file)
    # methods - various feature selection methods we outlined before, use them all here (list)
    
    # preprocessing
    X, y, num_feats = preprocess_dataset(dataset_path)
    
    # Run every method we outlined above from the methods list and collect returned best features from every method
    if 'pearson' in methods:
        cor_support, cor_feature = cor_selector(X, y,num_feats)
    if 'chi-square' in methods:
        chi_support, chi_feature = chi_squared_selector(X, y,num_feats)
    if 'rfe' in methods:
        rfe_support, rfe_feature = rfe_selector(X, y,num_feats)
    if 'log-reg' in methods:
        embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
    if 'rf' in methods:
        embedded_rf_support, embedded_rf_feature = embedded_rf_selector(X, y, num_feats)
    if 'lgbm' in methods:
        embedded_lgbm_support, embedded_lgbm_feature = embedded_lgbm_selector(X, y, num_feats)
    

    # Combine all the above feature list and count the maximum set of features that got selected by all methods
    #### Your Code starts here (Multiple lines)
    
    feature_name = list(X.columns)

    Features_selection=pd.DataFrame({'Features':feature_name,'Pearson': cor_support,'Chi-2':chi_support,'RFE':rfe_support,'Logistics':embedded_lr_support,'Random Forest':embedded_rf_support,'LightGBM':embedded_lgbm_support})
    Features_selection['Total']=np.sum(Features_selection,axis=1)
    Features_selection=Features_selection.sort_values(['Total','Features'] , ascending=False)
    Features_selection.index = range(1, len(Features_selection)+1)
    best_features = Features_selection['Features'].tolist()[:5]
    
    #### Your Code ends here
    return best_features,Features_selection



#  Helper function to plot Confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(figsize = (10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size = 18)
    plt.xlabel('Predicted label', size = 18)


def model_logistic(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y)
    lr=LogisticRegression(penalty='l2',random_state=42,n_jobs=-1)
    lr.fit(X_train,y_train)
    preds=lr.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    plot_confusion_matrix(cm, classes = ['Bad Customer', 'Good Customer'],
                      title = 'Confusion Matrix')
    accuracy=accuracy_score(y_test,preds)
    
    return accuracy

def model_randomforest(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y)
    rf=RandomForestClassifier(n_estimators=100, random_state=42, max_features = 'sqrt', n_jobs=-1, verbose = 1)
    rf.fit(X_train,y_train)
    pred1=rf.predict(X_test)
    cm = confusion_matrix(y_test, pred1)
    plot_confusion_matrix(cm, classes = ['Bad Customer', 'Good Customer'],
                      title = 'Confusion Matrix')
    accuracy=accuracy_score(y_test,pred1)

    return accuracy

def model_knn(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y)
    knn=KNeighborsClassifier()
    knn.fit(X_train,y_train)
    pred2=knn.predict(X_test)
    cm = confusion_matrix(y_test, pred2)
    plot_confusion_matrix(cm, classes = ['Bad Customer', 'Good Customer'],
                      title = 'Confusion Matrix')
    accuracy=accuracy_score(y_test,pred2)
    return accuracy

def model_svc(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y)
    svc=SVC()
    svc.fit(X_train,y_train)
    pred3=svc.predict(X_test)
    cm = confusion_matrix(y_test, pred3)
    plot_confusion_matrix(cm, classes = ['Bad Customer', 'Good Customer'],
                      title = 'Confusion Matrix')
    accuracy=accuracy_score(y_test,pred3)
    return accuracy

def model_nb(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y)
    nb=GaussianNB()
    nb.fit(X_train,y_train)
    pred4=nb.predict(X_test)
    cm = confusion_matrix(y_test, pred4)
    plot_confusion_matrix(cm, classes = ['Bad Customer', 'Good Customer'],
                      title = 'Confusion Matrix')
    accuracy=accuracy_score(y_test,pred4)   
    return accuracy

def stacking_model(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y)
    lr=LogisticRegression(penalty='l2',random_state=42,n_jobs=-1)
    rf=RandomForestClassifier(n_estimators=100, random_state=42, max_features = 'sqrt', n_jobs=-1, verbose = 1)  
    knn=KNeighborsClassifier()
    svc=SVC()    
    nb=GaussianNB()
    models = {'lr': lr,
          'knn': knn,
          'rf': rf,
          'svm': svc,
          'bayes': nb}
    stacking_model = StackingClassifier(estimators = [('lr', lr),
          ('knn', knn),
          ('rf', rf),
          ('svm', svc),
          ('bayes', nb)],
                                        final_estimator = lr,
                                        cv=5)
    def evaluate_model(model, X, y):
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores

    results = algorithms = []
    for algo, model in models.items():
        scores = evaluate_model(model, X, y)
        results.append(scores)
        algorithms.append(algo)
        print(f"Algorithm {algo}'s Accuracy >>> {np.mean(scores)} & Standard Deviation >>> {np.std(scores)}")
    
    algo = 'Stacking'
    scores = evaluate_model(stacking_model, X, y)
    print(f"Algorithm {algo}'s Accuracy >>> {np.mean(scores)} & Standard Deviation >>> {np.std(scores)}")
    stacking_model.fit(X_train, y_train)
    predict=stacking_model.predict(X_test)  
    cm = confusion_matrix(y_test, predict)
    plot_confusion_matrix(cm, classes = ['Bad Customer', 'Good Customer'],
                      title = 'Confusion Matrix')
    return
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
