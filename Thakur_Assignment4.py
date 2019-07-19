
# coding: utf-8

# In[104]:


# Boston Housing Study (Python)
#Prabhat Thakur  Date 10/21/2018
#MSDS422 - Assignment4 - Adding RandomForestRegressor


# In[105]:


# import base packages into the namespace for this program
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
# modeling routines from Scikit Learn packages
import sklearn.linear_model 
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score  
from sklearn.ensemble import RandomForestRegressor

from math import sqrt  # for root mean-squared error calculation


# read data for the Boston Housing Study
# creating data frame restdata
boston_input = pd.read_csv('boston.csv')

# check the pandas DataFrame object boston_input
print('\nboston DataFrame (first and last five rows):')
print(boston_input.head())
print(boston_input.tail())

print('\nGeneral description of the boston_input DataFrame:')
print(boston_input.info())

# drop neighborhood from the data being considered
boston = boston_input.drop('neighborhood', 1)
print('\nGeneral description of the boston DataFrame:')
print(boston.info())

print('\nDescriptive statistics of the boston DataFrame:')
print(boston.describe())

# set up preliminary data for data for fitting the models 
# the first column is the median housing value response
# the remaining columns are the explanatory variables
prelim_model_data = np.array([boston.mv,    boston.crim,    boston.zn,    boston.indus,    boston.chas,    boston.nox,    boston.rooms,    boston.age,    boston.dis,    boston.rad,    boston.tax,    boston.ptratio,    boston.lstat]).T

# dimensions of the polynomial model X input and y response
# preliminary data before standardization
print('\nData dimensions:', prelim_model_data.shape)

# standard scores for the columns... along axis 0
scaler = StandardScaler()
print(scaler.fit(prelim_model_data))
# show standardization constants being employed
print(scaler.mean_)
print(scaler.scale_)

# the model data will be standardized form of preliminary model data
model_data = scaler.fit_transform(prelim_model_data)

# dimensions of the polynomial model X input and y response
# all in standardized units of measure
print('\nDimensions for model_data:', model_data.shape)


# In[106]:


boston.hist(bins=50, figsize=(20,15))
plt.show()


# In[107]:


corr_matrix = boston.corr()
corr_matrix["mv"].sort_values(ascending=False)


# In[108]:


#let’s just focus on a few promising attributes that seem most correlated with the Median value of homes
from pandas.tools.plotting import scatter_matrix
attributes = ["mv", "rooms", "zn","nox", "tax","indus","ptratio","lstat"]
scatter_matrix(boston[attributes], figsize=(16, 12))


# In[109]:


#The most promising attribute to predict the median house value is the rooms and istat and nox.
#so let’s zoom in on their correlation scatterplot:
boston.plot(kind="scatter", x="rooms", y="mv", alpha=0.4)
boston.plot(kind="scatter", x="lstat", y="mv", alpha=0.3)
boston.plot(kind="scatter", x="nox", y="mv", alpha=0.2)


# In[110]:


# seed value for random number generators to obtain reproducible results
RANDOM_SEED = 1

# although we standardize X and y variables on input,
# we will fit the intercept term in the models
# Expect fitted values to be close to zero
SET_FIT_INTERCEPT = True

def modelstotest(option):
    if option == 'lr':
        names = ['Linear_Regression', 'Ridge_Regression', 'Lasso_Regression', 
         'ElasticNet_Regression'] 

        regressors = [LinearRegression(fit_intercept = SET_FIT_INTERCEPT),
                  Ridge(alpha = 1, solver = 'cholesky', fit_intercept = SET_FIT_INTERCEPT, normalize = False, 
                    random_state = RANDOM_SEED),
                  Lasso(alpha = 0.1, max_iter=10000, tol=0.01, fit_intercept = SET_FIT_INTERCEPT, 
                    random_state = RANDOM_SEED),
                  ElasticNet(alpha = 0.1, l1_ratio = 0.5, max_iter=10000, tol=0.01, fit_intercept = SET_FIT_INTERCEPT, 
                         normalize = False, random_state = RANDOM_SEED)]
        
    
    if option == 'lrrfr':
        names = ['Linear_Regression', 'Ridge_Regression', 'Lasso_Regression', 
         'ElasticNet_Regression','RandomForest_Regressor'] 

        regressors = [LinearRegression(fit_intercept = SET_FIT_INTERCEPT),
              Ridge(alpha = 1, solver = 'cholesky', fit_intercept = SET_FIT_INTERCEPT, normalize = False, 
                    random_state = RANDOM_SEED),
              Lasso(alpha = 0.1, max_iter=10000, tol=0.01, fit_intercept = SET_FIT_INTERCEPT, 
                    random_state = RANDOM_SEED),
              ElasticNet(alpha = 0.1, l1_ratio = 0.5, max_iter=10000, tol=0.01, fit_intercept = SET_FIT_INTERCEPT, 
                         normalize = False, random_state = RANDOM_SEED),
              RandomForestRegressor( max_features = 5, n_estimators=500, bootstrap=True, n_jobs=-1,random_state = RANDOM_SEED)]
        
    
    return names,regressors


# In[111]:


model_data_features = model_data[:, 1:model_data.shape[1]]
model_data_label = model_data[:, 0]

names,regressors = modelstotest('lr')

for name, reg_model in zip(names,regressors):
    print('\nRegression model evaluation for:', name)
    print('  Scikit Learn method:', reg_model)
    reg_model.fit(model_data_features, model_data_label)  # fit on the entire set.
    print('Fitted regression intercept:', reg_model.intercept_)
    print('Fitted regression coefficients:', reg_model.coef_)
 
    # evaluate on model
    model_data_predict = reg_model.predict(model_data_features)
    print('Coefficient of determination (R-squared):', r2_score(model_data_label, model_data_predict))
    method_result = sqrt(mean_squared_error(model_data_label, model_data_predict))
    print(reg_model.get_params(deep=True))
    print('Root mean-squared error:', method_result)
    


# In[112]:


reg_model  = RandomForestRegressor( max_features = 5, n_estimators=500, bootstrap=True, n_jobs=-1,random_state = RANDOM_SEED)
print('\nRegression model evaluation for:', 'RandomForest_Regressor', '- with feature scalling')
print('  Scikit Learn method:', reg_model)
reg_model.fit(model_data_features, model_data_label)  # fit on the entire set.
 
# evaluate on model
model_data_predict = reg_model.predict(model_data_features)
print('Coefficient of determination (R-squared):', r2_score(model_data_label, model_data_predict))
method_result = sqrt(mean_squared_error(model_data_label, model_data_predict))
print(reg_model.get_params(deep=True))
print('Root mean-squared error:', method_result)

model_data_features_nfs = prelim_model_data[:, 1:prelim_model_data.shape[1]]
model_data_label_nfs = prelim_model_data[:, 0]
print('\n----------------------------------------------------------------------')
print(' Regression model evaluation for:', 'RandomForest_Regressor', '- without feature scalling')
print('  Scikit Learn method:', reg_model)
reg_model.fit(model_data_features_nfs, model_data_label_nfs)  # fit on the entire set.
 
# evaluate on model
model_data_predict_nfs = reg_model.predict(model_data_features_nfs)
print('Coefficient of determination (R-squared):', r2_score(model_data_label_nfs, model_data_predict_nfs))
method_result_nfs = sqrt(mean_squared_error(model_data_label_nfs, model_data_predict_nfs))
print(reg_model.get_params(deep=True))
print('Root mean-squared error:', method_result_nfs)


# In[113]:


# --------------------------------------------------------
# specify the k-fold cross-validation design
from sklearn.model_selection import KFold

# ten-fold cross-validation employed here
# As an alternative to 10-fold cross-validation, restdata with its 
# small sample size could be analyzed would be a good candidate
# for  leave-one-out cross-validation, which would set the number
# of folds to the number of observations in the data set.
N_FOLDS = 10

names,regressors = modelstotest('lrrfr')

# set up numpy array for storing results
cv_results = np.zeros((N_FOLDS, len(names)))

kf = KFold(n_splits = N_FOLDS, shuffle=False, random_state = RANDOM_SEED)
# check the splitting process by looking at fold observation counts
index_for_fold = 0  # fold count initialized 
for train_index, test_index in kf.split(model_data):
    print('\nFold index:', index_for_fold,
          '------------------------------------------')
    #   the structure of modeling data for this study has the
    #   response variable coming first and explanatory variables later          
    #   so 1:model_data.shape[1] slices for explanatory variables
    #   and 0 is the index for the response variable    
    X_train = model_data[train_index, 1:model_data.shape[1]]
    X_test = model_data[test_index, 1:model_data.shape[1]]
    y_train = model_data[train_index, 0]
    y_test = model_data[test_index, 0]   
    print('\nShape of input data for this fold:',
          '\nData Set: (Observations, Variables)')
    print('X_train:', X_train.shape)
    print('X_test:',X_test.shape)
    print('y_train:', y_train.shape)
    print('y_test:',y_test.shape)
    
    index_for_method = 0  # initialize
    for name, reg_model in zip(names, regressors):
        print('\nRegression model evaluation for:', name)
        print('  Scikit Learn method:', reg_model)
        reg_model.fit(X_train, y_train)  # fit on the train set for this fold
 
        # evaluate on the test set for this fold
        y_test_predict = reg_model.predict(X_test)
        fold_method_result = sqrt(mean_squared_error(y_test, y_test_predict))
        print(reg_model.get_params(deep=True))
        print('Root mean-squared error:', fold_method_result)
        cv_results[index_for_fold, index_for_method] = fold_method_result
        index_for_method += 1
  
    index_for_fold += 1

cv_results_df = pd.DataFrame(cv_results)
cv_results_df.columns = names

print('\n----------------------------------------------')
print('Results from ', N_FOLDS, '-fold cross-validation\n')
print(cv_results_df.describe()) 

print('\n----------------------------------------------')
print('Average results from ', N_FOLDS, '-fold cross-validation\n',
      'in standardized units (mean 0, standard deviation 1)\n',
      '\nMethod               Root mean-squared error', sep = '')     
 
print(cv_results_df.mean())  

 


# In[114]:


# Use GridSearch view to find Best Parameters for all 4 models
names,regressors = modelstotest('lr')
from sklearn.model_selection import GridSearchCV
param_grid = [
 {'fit_intercept': [True,False], 'normalize': [True,False]}]
for name, reg_model in zip(names, regressors):
    print('\nRegression model evaluation for:', name)
    #print('  Scikit Learn method:', reg_model)
    grid_search = GridSearchCV(reg_model, param_grid, cv=10, scoring='neg_mean_squared_error')
    grid_search.fit(model_data_features, model_data_label)  
    print('Best Parameters:\n', grid_search.best_params_)
    print('Best Estimatores:\n',grid_search.best_estimator_)


# In[115]:


#Grid Search for RandomForestRegressor
param_grid = [
 {'n_estimators': [100,200, 300,400, 500], 'max_features': [2, 4, 6, 8,10]} ]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=10,
 scoring='neg_mean_squared_error')
grid_search.fit(model_data_features_nfs, model_data_label_nfs)
print('Best Parameters:\n', grid_search.best_params_)
print('Best Estimatores:\n',grid_search.best_estimator_)


# In[116]:


cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# In[117]:


reg_model  = RandomForestRegressor( max_features = 6, n_estimators=400, bootstrap=True, n_jobs=-1,random_state = RANDOM_SEED)

print('\n----------------------------------------------------------------------')
print(' Regression model evaluation for:', 'RandomForest_Regressor', '- without feature scalling')
print('  Scikit Learn method:', reg_model)
reg_model.fit(model_data_features_nfs, model_data_label_nfs)  # fit on the entire set.
 
# evaluate on model
model_data_predict_nfs = reg_model.predict(model_data_features_nfs)
print('Coefficient of determination (R-squared):', r2_score(model_data_label_nfs, model_data_predict_nfs))
method_result_nfs = sqrt(mean_squared_error(model_data_label_nfs, model_data_predict_nfs))
print(reg_model.get_params(deep=True))
print('Root mean-squared error:', method_result_nfs)


# In[118]:


# K-Fold  cross-validation design for RandomForestRegressor with best parameters.
# Using Normal and Scalled features.

reg_model  = RandomForestRegressor( max_features = 6, n_estimators=400, bootstrap=True, n_jobs=-1,random_state = RANDOM_SEED)

# set up numpy array for storing results
cv_results = np.zeros((N_FOLDS,2))

kf = KFold(n_splits = N_FOLDS, shuffle=False, random_state = RANDOM_SEED)
# check the splitting process by looking at fold observation counts
index_for_fold = 0  # fold count initialized 
for train_index, test_index in kf.split(prelim_model_data):
    print('\nFold index:', index_for_fold,
          '------------------------------------------')
    #   the structure of modeling data for this study has the
    #   response variable coming first and explanatory variables later          
    #   so 1:model_data.shape[1] slices for explanatory variables
    #   and 0 is the index for the response variable    
    X_train = prelim_model_data[train_index, 1:prelim_model_data.shape[1]]
    X_test = prelim_model_data[test_index, 1:prelim_model_data.shape[1]]
    y_train = prelim_model_data[train_index, 0]
    y_test = prelim_model_data[test_index, 0]   
    
    X_train1 = model_data[train_index, 1:model_data.shape[1]]
    X_test1 = model_data[test_index, 1:model_data.shape[1]]
    y_train1 = model_data[train_index, 0]
    y_test1 = model_data[test_index, 0]   
    
    print('\nShape of input data for this fold:','\nData Set: (Observations, Variables)')
    print('X_train:', X_train.shape)
    print('X_test:',X_test.shape)
    print('y_train:', y_train.shape)
    print('y_test:',y_test.shape)
    
    reg_model.fit(X_train, y_train)  # fit on the train set for this fold
 
    # evaluate on the test set for this fold
    y_test_predict = reg_model.predict(X_test)
    fold_method_result = sqrt(mean_squared_error(y_test, y_test_predict))
    #print(reg_model.get_params(deep=True))
    print('Root mean-squared error:', fold_method_result)
    cv_results[index_for_fold,0] = fold_method_result
    
    reg_model.fit(X_train1, y_train1)  # fit on the train set for this fold
    # evaluate on the test set for this fold
    y_test_predict1 = reg_model.predict(X_test1)
    fold_method_result = sqrt(mean_squared_error(y_test1, y_test_predict1))
    #print(reg_model.get_params(deep=True))
    print('Root mean-squared error with Scalling:', fold_method_result)
    cv_results[index_for_fold,1] = fold_method_result
  
    index_for_fold += 1

cv_results_df = pd.DataFrame(cv_results)
cv_results_df.columns = ['RandomForest_Regressor','RandomForest_RegressorwithScalling']

print('\n----------------------------------------------')
print('Results from ', N_FOLDS, '-fold cross-validation\n')
print(cv_results_df.describe()) 

print('\n----------------------------------------------')
print('Average results from ', N_FOLDS, '-fold cross-validation\n',
      'in standardized units (mean 0, standard deviation 1)\n',
      '\nMethod               Root mean-squared error', sep = '')     
 
print(cv_results_df.mean())  


# In[119]:


attributes = boston.columns
feature_importances = grid_search.best_estimator_.feature_importances_
print ('feature_importances :\n',feature_importances)
sorted(zip(feature_importances, attributes), reverse=True)

