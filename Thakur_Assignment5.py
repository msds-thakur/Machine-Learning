
# coding: utf-8

# In[1]:


#Prabhat Thakur  Date 11/04/2018
#MSDS422 - Assignment-5 
#MNIST dataset benchmark testing for RandomForestClassifier before and after using PCA. 


# In[2]:


# import base packages into the namespace for this program
from scipy.io import loadmat
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

import matplotlib
import matplotlib.pyplot as plt

import csv
import time
import numpy as np
import pandas as pd


# In[3]:


# Load MNIST dataset from .mat file.

mnist_raw = loadmat("mnist-original.mat")
mnist = {
"data": mnist_raw["data"].T,
"target": mnist_raw["label"][0],
"COL_NAMES": ["label", "data"],
"DESCR": "mldata.org dataset: mnist-original"}

print('mnist dataset \n', mnist)
X, y = mnist['data'], mnist['target']
print('\ndata shape: ', X.shape)
print('target shape: ', y.shape)


# In[4]:


# Display 38,001th image
some_digit = X[38001]
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, 
           interpolation = 'nearest')
plt.axis('on')
plt.show()
print('\n y[38001] = ',y[38001])


# In[5]:


# Split the data to 60,000 images as training data and 10,000 as test data
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# --- Random Forest Classifier ---
rfClf = RandomForestClassifier(bootstrap = True,n_estimators=100, max_leaf_nodes=10, n_jobs=-1,random_state =42)    

# Record the time it takes to fit the model

# set random number seed 

np.random.seed(seed = 9999)

replications = 10  # repeat the trial ten times
x_time = [] # empty list for storing test results
n = 0  # initialize count
print('--- Time to Fit Random Forest Classifier ---')

while (n < replications): 
    start_time = time.clock()
    # generate 1 million random negative binomials and store in a vector
    rfClf.fit(X_train, y_train)
    end_time = time.clock()
    runtime = end_time - start_time  # seconds of wall-clock time
    x_time.append(runtime * 1000)  # report in milliseconds 
    print("replication", n + 1, ":", x_time[n], "milliseconds") 
    n = n + 1

# write results to external file 
with open('rf_fit.csv', 'wt') as f:
    writer = csv.writer(f, quoting = csv.QUOTE_NONNUMERIC, dialect = 'excel')
    writer.writerow('x_time')    
    for i in range(replications):
        writer.writerow(([x_time[i],]))

# preliminary analysis for this cell of the design
print(pd.DataFrame(x_time).describe())

y_predict = rfClf.predict(X_test)

# Performance measurement using F1-score
f1Score = f1_score(y_test, y_predict, average='weighted')
print('\nF1 Score: ', f1Score)


# In[6]:


conf_mx = confusion_matrix(y_test, y_predict)
print ('Confusion Matrix :\n',conf_mx)

plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()

row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums

np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()


# In[7]:


# Split the data to 60,000 images as training data and 10,000 as test data using train test split
from sklearn.model_selection import train_test_split

X_train1, X_test1, y_train1, y_test1 = train_test_split(X,y,test_size =1/7,random_state =42)

# --- Random Forest Classifier with bootstrap = True and max_features = 'sqrt'---
rfClf1 = RandomForestClassifier(bootstrap = True, max_features = 'sqrt', n_estimators=100, max_leaf_nodes=10, n_jobs=-1,random_state =42)    

# Record the time it takes to fit the model

# set random number seed 

np.random.seed(seed = 9999)

replications = 10  # repeat the trial ten times
x_time = [] # empty list for storing test results
n = 0  # initialize count
print('--- Time to Fit Random Forest Classifier ---')

while (n < replications): 
    start_time = time.clock()
    # generate 1 million random negative binomials and store in a vector
    rfClf1.fit(X_train1, y_train1)
    end_time = time.clock()
    runtime = end_time - start_time  # seconds of wall-clock time
    x_time.append(runtime * 1000)  # report in milliseconds 
    print("replication", n + 1, ":", x_time[n], "milliseconds") 
    n = n + 1

# preliminary analysis for this cell of the design
print(pd.DataFrame(x_time).describe())

y_predict1 = rfClf1.predict(X_test1)

# Performance measurement using F1-score
f1Score1 = f1_score(y_test1, y_predict1, average='weighted')
print('\nF1 Score on test set: ', f1Score1)


# In[8]:


conf_mx1 = confusion_matrix(y_test1, y_predict1)
print ('Confusion Matrix :\n',conf_mx)

plt.matshow(conf_mx1, cmap=plt.cm.gray)
plt.show()

row_sums = conf_mx1.sum(axis=1, keepdims=True)
norm_conf_mx1 = conf_mx1 / row_sums

np.fill_diagonal(norm_conf_mx1, 0)
plt.matshow(norm_conf_mx1, cmap=plt.cm.gray)
plt.show()


# In[9]:


# --- PCA ---
# Generate principal components that represent 95 percent of the variability
# in the explanatory variables
pca = PCA(n_components=0.95)

# Runtime to identify the principal components
pca_time = [] # empty list for storing test results
n = 0  # initialize count
print('--- Time to Identify Pricipal Components ---')
while (n < replications): 
    start_time = time.clock()
    # generate 1 million random negative binomials and store in a vector
    X_pca = pca.fit_transform(X) # run on all 70,000 observations
    end_time = time.clock()
    runtime = end_time - start_time  # seconds of wall-clock time
    pca_time.append(runtime * 1000)  # report in milliseconds 
    print("replication", n + 1, ":", pca_time[n], "milliseconds") 
    n = n + 1

# write results to external file 
with open('pca_fit.csv', 'wt') as f:
    writer = csv.writer(f, quoting = csv.QUOTE_NONNUMERIC, dialect = 'excel')
    writer.writerow('pca_time')    
    for i in range(replications):
        writer.writerow(([pca_time[i],]))

print(pd.DataFrame(pca_time).describe())

# show summary of pca solution
pca_explained_variance = pca.explained_variance_ratio_
print('\nPrincipal components count:', len(pca_explained_variance))
# -- Results: Dimension is reduced to 154 variables from 784 variables
print('\nProportion of variance explained:', pca_explained_variance)


# In[10]:


# Split the reduced data to 60,000 images as training data and 10,000 as test data
X_pca_train, X_pca_test = X_pca[:60000], X_pca[60000:]

# Random Forest Classifier using the principal components
rfClf_pca = RandomForestClassifier(bootstrap = True, n_estimators=100, max_leaf_nodes=10, n_jobs=-1,random_state =42)

# Runtime to identify the principal components
x_pca_time = [] # empty list for storing test results
n = 0  # initialize count
print('--- Time to Fit Random Forest Classifier using Principal Components ---')
while (n < replications): 
    start_time = time.clock()
    # generate 1 million random negative binomials and store in a vector
    rfClf_pca.fit(X_pca_train, y_train)
    end_time = time.clock()
    runtime = end_time - start_time  # seconds of wall-clock time
    x_pca_time.append(runtime * 1000)  # report in milliseconds 
    print("replication", n + 1, ":", x_pca_time[n], "milliseconds") 
    n = n + 1

# write results to external file 
with open('rf_pca_fit.csv', 'wt') as f:
    writer = csv.writer(f, quoting = csv.QUOTE_NONNUMERIC, dialect = 'excel')
    writer.writerow('x_pca_time')    
    for i in range(replications):
        writer.writerow(([x_pca_time[i],]))
        
print(pd.DataFrame(x_pca_time).describe())

y_predict_pca = rfClf_pca.predict(X_pca_test)

# Performance measurement using F1-score

f1Score_pca = f1_score(y_test, y_predict_pca, average='weighted')
print('\nF1 Score: ', f1Score_pca)


# In[11]:


conf_mx1 = confusion_matrix(y_test1, y_predict1)
print ('Confusion Matrix :\n',conf_mx)

plt.matshow(conf_mx1, cmap=plt.cm.gray)
plt.show()

row_sums = conf_mx1.sum(axis=1, keepdims=True)
norm_conf_mx1 = conf_mx1 / row_sums

np.fill_diagonal(norm_conf_mx1, 0)
plt.matshow(norm_conf_mx1, cmap=plt.cm.gray)
plt.show()


# In[12]:


# --- PCA on Training Data only (60000 observations)---
# Generate principal components that represent 95 percent of the variability
# in the explanatory variables.
pca = PCA(n_components=0.95)

# Runtime to identify the principal components
pca_time = [] # empty list for storing test results
n = 0  # initialize count
print('--- Time to Identify Pricipal Components ---\n')
while (n < replications): 
    start_time = time.clock()
    X_pca_train = pca.fit_transform(X_train) # run on 60,000 observations
    end_time = time.clock()
    runtime = end_time - start_time  # seconds of wall-clock time
    pca_time.append(runtime * 1000)  # report in milliseconds 
    print("replication", n + 1, ":", pca_time[n], "milliseconds") 
    n = n + 1

print(pd.DataFrame(pca_time).describe())

# show summary of pca solution
pca_explained_variance = pca.explained_variance_ratio_
print('\nPrincipal components count:', len(pca_explained_variance))
# -- Results: Dimension is reduced to 154 variables from 784 variables

X_pca_test = pca.transform(X_test)


# In[13]:


# Random Forest Classifier using the principal components
rfClf_pca = RandomForestClassifier(bootstrap = True, n_estimators=100, max_leaf_nodes=10, n_jobs=-1,random_state =42)

# Runtime to identify the principal components
x_pca_time = [] # empty list for storing test results
n = 0  # initialize count
print('--- Time to Fit Random Forest Classifier using Principal Components ---\n')
while (n < replications): 
    start_time = time.clock()
    # generate 1 million random negative binomials and store in a vector
    rfClf_pca.fit(X_pca_train, y_train)
    end_time = time.clock()
    runtime = end_time - start_time  # seconds of wall-clock time
    x_pca_time.append(runtime * 1000)  # report in milliseconds 
    print("replication", n + 1, ":", x_pca_time[n], "milliseconds") 
    n = n + 1
       
print(pd.DataFrame(x_pca_time).describe())

y_predict_pca = rfClf_pca.predict(X_pca_test)


# In[14]:


# Performance measurement using F1-score
f1Score_pca = f1_score(y_test, y_predict_pca, average='weighted')
print('\nF1 Score: ', f1Score_pca)

conf_mx1 = confusion_matrix(y_test1, y_predict1)
print ('Confusion Matrix :\n',conf_mx)

plt.matshow(conf_mx1, cmap=plt.cm.gray)
plt.show()

row_sums = conf_mx1.sum(axis=1, keepdims=True)
norm_conf_mx1 = conf_mx1 / row_sums

np.fill_diagonal(norm_conf_mx1, 0)
plt.matshow(norm_conf_mx1, cmap=plt.cm.gray)
plt.show()

