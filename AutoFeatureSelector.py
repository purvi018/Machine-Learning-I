#!/usr/bin/env python
# coding: utf-8

# # Task 7: AutoFeatureSelector Tool
# ## This task is to test your understanding of various Feature Selection methods outlined in the lecture and the ability to apply this knowledge in a real-world dataset to select best features and also to build an automated feature selection tool as your toolkit
# 
# ### Use your knowledge of different feature selector methods to build an Automatic Feature Selection tool
# - Pearson Correlation
# - Chi-Square
# - RFE
# - Embedded
# - Tree (Random Forest)
# - Tree (Light GBM)

# ### Dataset: FIFA 19 Player Skills
# #### Attributes: FIFA 2019 players attributes like Age, Nationality, Overall, Potential, Club, Value, Wage, Preferred Foot, International Reputation, Weak Foot, Skill Moves, Work Rate, Position, Jersey Number, Joined, Loaned From, Contract Valid Until, Height, Weight, LS, ST, RS, LW, LF, CF, RF, RW, LAM, CAM, RAM, LM, LCM, CM, RCM, RM, LWB, LDM, CDM, RDM, RWB, LB, LCB, CB, RCB, RB, Crossing, Finishing, Heading, Accuracy, ShortPassing, Volleys, Dribbling, Curve, FKAccuracy, LongPassing, BallControl, Acceleration, SprintSpeed, Agility, Reactions, Balance, ShotPower, Jumping, Stamina, Strength, LongShots, Aggression, Interceptions, Positioning, Vision, Penalties, Composure, Marking, StandingTackle, SlidingTackle, GKDiving, GKHandling, GKKicking, GKPositioning, GKReflexes, and Release Clause.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss
from collections import Counter
import math
from scipy import stats


# In[2]:


player_df = pd.read_csv("fifa19.csv")


# In[3]:


numcols = ['Overall', 'Crossing','Finishing',  'ShortPassing',  'Dribbling','LongPassing', 'BallControl', 'Acceleration','SprintSpeed', 'Agility',  'Stamina','Volleys','FKAccuracy','Reactions','Balance','ShotPower','Strength','LongShots','Aggression','Interceptions']
catcols = ['Preferred Foot','Position','Body Type','Nationality','Weak Foot']


# In[4]:


player_df = player_df[numcols+catcols]


# In[5]:


traindf = pd.concat([player_df[numcols], pd.get_dummies(player_df[catcols])],axis=1)
features = traindf.columns

traindf = traindf.dropna()


# In[6]:


traindf = pd.DataFrame(traindf,columns=features)


# In[7]:


y = traindf['Overall']>=87
X = traindf.copy()
del X['Overall']


# In[8]:


X.head()


# In[9]:


len(X.columns)


# ### Set some fixed set of features

# In[15]:


feature_name = list(X.columns)
# no of maximum features we need to select
num_feats=30


# ## Filter Feature Selection - Pearson Correlation

# ### Pearson Correlation function

# In[21]:


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


# In[22]:


cor_support, cor_features = cor_selector(X, y,30)
print(str(len(cor_features)), 'selected features')


# ### List the selected features from Pearson Correlation

# In[24]:


cor_features


# ## Filter Feature Selection - Chi-Sqaure

# In[25]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler


# ### Chi-Squared Selector function

# In[111]:


def chi_squared_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    Xscale=MinMaxScaler().fit_transform(X)
    selector=SelectKBest(chi2,num_feats)
    selector.fit(Xscale,y)
    chi_score=selector.scores_
    chi_support=selector.get_support()
    chi_features=X.loc[:,chi_support].columns.tolist()
    # Your code ends here
    return chi_support, chi_features


# In[112]:


chi_support, chi_features = chi_squared_selector(X, y,num_feats)
print(str(len(chi_features)), 'selected features')


# ### List the selected features from Chi-Square 

# In[113]:


chi_features


# ## Wrapper Feature Selection - Recursive Feature Elimination

# In[35]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


# ### RFE Selector function

# In[41]:


def rfe_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    Xscale=MinMaxScaler().fit_transform(X)
    rfe_selector=RFE(estimator=LogisticRegression(),n_features_to_select=num_feats,step=10,verbose=1)
    rfe_selector.fit(Xscale,y)
    rfe_support=rfe_selector.get_support()
    rfe_feature=X.loc[:,rfe_support].columns.tolist()
    # Your code ends here
    return rfe_support, rfe_feature


# In[42]:


rfe_support, rfe_feature = rfe_selector(X, y,num_feats)
print(str(len(rfe_feature)), 'selected features')


# ### List the selected features from RFE

# In[43]:


rfe_feature


# ## Embedded Selection - Lasso: SelectFromModel

# In[44]:


from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


# In[50]:


def embedded_log_reg_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    
    Xscale=MinMaxScaler().fit_transform(X)
    embedded_lr_selector=SelectFromModel(estimator=LogisticRegression(),max_features=num_feats)
    embedded_lr_selector.fit(Xscale,y)
    embedded_lr_support=embedded_lr_selector.get_support()
    embedded_lr_feature=X.loc[:,embedded_lr_support].columns.tolist()    
    
    # Your code ends here
    return embedded_lr_support, embedded_lr_feature


# In[51]:


num_feats=30
embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
print(str(len(embedded_lr_feature)), 'selected features')


# In[52]:


embedded_lr_feature


# ## Tree based(Random Forest): SelectFromModel

# In[106]:


from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier


# In[107]:


def embedded_rf_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
   
    embedded_rf_selector=SelectFromModel(estimator=RandomForestClassifier(n_estimators=50),max_features=num_feats)
    embedded_rf_selector.fit(X,y)
    embedded_rf_support=embedded_rf_selector.get_support()
    embedded_rf_feature=X.loc[:,embedded_rf_support].columns.tolist()     
    # Your code ends here
    return embedded_rf_support, embedded_rf_feature


# In[108]:


#num_feats=30
embedded_rf_support, embedded_rf_feature = embedded_rf_selector(X, y, num_feats)
print(str(len(embedded_rf_feature)), 'selected features')


# In[66]:


embedded_rf_feature


# ## Tree based(Light GBM): SelectFromModel

# In[67]:


from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier


# In[92]:


def embedded_lgbm_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    
    embedded_lgbm_selector=SelectFromModel(estimator=LGBMClassifier(n_estimators=500,num_leaves=32,learning_rate=0.05,
    colsample_bytree=0.2, reg_alpha=3.0,reg_lambda=0.0,min_split_gain=0.01,min_child_weight=40)
                                           ,max_features=num_feats)
    embedded_lgbm_selector.fit(X,y)
    embedded_lgbm_support=embedded_lgbm_selector.get_support()
    embedded_lgbm_feature=X.loc[:,embedded_lgbm_support].columns.tolist()     
    # Your code ends here
    return embedded_lgbm_support, embedded_lgbm_feature


# In[93]:


embedded_lgbm_support, embedded_lgbm_feature = embedded_lgbm_selector(X, y, num_feats)
print(str(len(embedded_lgbm_feature)), 'selected features')


# In[94]:


embedded_lgbm_feature


# ## Putting all of it together: AutoFeatureSelector Tool

# In[129]:


#pd.set_option('display.max_rows', None)
# put all selection together
feature_selection_df = pd.DataFrame({'Feature':feature_name, 'Pearson':cor_support, 'Chi-2':chi_support, 'RFE':rfe_support, 'Logistics':embedded_lr_support,'Random Forest':embedded_rf_support, 'LightGBM':embedded_lgbm_support})
# count the selected times for each feature
feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
# display the top 100
feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
feature_selection_df.index = range(1, len(feature_selection_df)+1)
feature_selection_df.head(num_feats)


# In[ ]:


best_features = feature_selection_df['Feature'].tolist()[:5]


# ## Can you build a Python script that takes dataset and a list of different feature selection methods that you want to try and output the best (maximum votes) features from all methods?

# In[117]:


def preprocess_dataset(dataset_path):
    # Your code starts here (Multiple lines)
    player_df = pd.read_csv("fifa19.csv")
    numcols = ['Overall', 'Crossing','Finishing',  'ShortPassing',  'Dribbling','LongPassing', 'BallControl', 'Acceleration','SprintSpeed', 'Agility',  'Stamina','Volleys','FKAccuracy','Reactions','Balance','ShotPower','Strength','LongShots','Aggression','Interceptions']
    catcols = ['Preferred Foot','Position','Body Type','Nationality','Weak Foot']
    player_df = player_df[numcols+catcols]
    traindf = pd.concat([player_df[numcols], pd.get_dummies(player_df[catcols])],axis=1)
    features = traindf.columns
    traindf = traindf.dropna()
    traindf = pd.DataFrame(traindf,columns=features)
    y = traindf['Overall']>=87
    X = traindf.copy()
    del X['Overall']
    num_feats=30
    feature_name = list(X.columns)
    # Your code ends here
    return X, y, num_feats


# In[137]:


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
    
    
    Features_selection=pd.DataFrame({'Features':feature_name,'Pearson': cor_support,'Chi-2':chi_support,'RFE':rfe_support,'Logistics':embedded_lr_support,'Random Forest':embedded_rf_support,'LightGBM':embedded_lgbm_support})
    Features_selection['Total']=np.sum(Features_selection,axis=1)
    Features_selection=Features_selection.sort_values(['Total','Features'] , ascending=False)
    Features_selection.index = range(1, len(Features_selection)+1)
    best_features = Features_selection['Features'].tolist()[:5]
    
    #### Your Code ends here
    return best_features


# In[119]:


X, y, num_feats=preprocess_dataset('fifa19.csv')
X, y, num_feats


# In[139]:


best_features = autoFeatureSelector(dataset_path='fifa19.csv', methods=['pearson', 'chi-square', 'rfe', 'log-reg', 'rf', 'lgbm'])
best_features


# ### Last, Can you turn this notebook into a python script, run it and submit the python (.py) file that takes dataset and list of methods as inputs and outputs the best features

# In[ ]:




