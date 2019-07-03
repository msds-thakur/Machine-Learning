
# coding: utf-8

# In[11]:


# Prabhat Thakur  Date 11/11/2018
# MSDS422 - Assignment-6 
# Demonstration of Benchmark Experiment using Scikit Learn for Artificial Neural Networks
# Utilizes the MNIST data. Completely crossed 3x3 benchmark experiment.

# Code reused from:
# 4_mnist_from_scratch-data-dump.py - For importing MNIST data.
# 6_mnist_from_scratch_scikit-learn-ann-v001.py -
# Demonstration of a completely crossed 2x2 benchmark experiment using Scikit Learn to build artificial neural networks.


# In[12]:


# coding: utf-8
# ensure common functions across Python 2 and 3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# MNIST from scratch (data and partitioning from Google tensorflow container)
# source:  https://hub.docker.com/r/tensorflow/tensorflow/
# 
# We begin with a notebook that walks through an example of training a TensorFlow model 
# to do digit classification using the [MNIST data set](http://yann.lecun.com/exdb/mnist/). 
# MNIST is a labeled set of images of handwritten digits.

import gzip, binascii, struct, numpy

# We'll proceed in steps, beginning with importing and inspecting the MNIST data. This doesn't have anything to do with TensorFlow in particular -- we're just downloading the data archive.
import os
from six.moves.urllib.request import urlretrieve

SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
WORK_DIRECTORY = "/tmp"

def maybe_download(filename):
    """A helper to download the data files if not present."""
    if not os.path.exists(WORK_DIRECTORY):
        os.mkdir(WORK_DIRECTORY)
    filepath = os.path.join(WORK_DIRECTORY, filename)
    if not os.path.exists(filepath):
        filepath, _ = urlretrieve(SOURCE_URL + filename, filepath)
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    else:
        print('Already downloaded', filename)
    return filepath

train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

    
IMAGE_SIZE = 28
PIXEL_DEPTH = 255

def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    For MNIST data, the number of channels is always 1.
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        # Skip the magic number and dimensions; we know these values.
        bytestream.read(16)

        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
        return data

train_data = extract_data(train_data_filename, 60000)
test_data = extract_data(test_data_filename, 10000)


# A crucial difference here is how we `reshape` the array of pixel values. 
#Instead of one image that's 28x28, we now have a set of 60,000 images, each one being 28x28. We also include a number 
#of channels, which for grayscale images as we have here is 1.
# 
print('Training data shape', train_data.shape)
# Looks good. Now we know how to index our full set of training and test images.

# ### Label data
# Let's move on to loading the full set of labels. As is typical in classification problems, we'll convert our input labels 
#into a [1-hot](https://en.wikipedia.org/wiki/One-hot) encoding over a length 10 vector corresponding to 10 digits. 
#The vector [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], for example, would correspond to the digit 1.

NUM_LABELS = 10

def extract_labels(filename, num_images):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        # Skip the magic number and count; we know these values.
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    # Convert to dense 1-hot representation.
    return (numpy.arange(NUM_LABELS) == labels[:, None]).astype(numpy.float32)

train_labels = extract_labels(train_labels_filename, 60000)
test_labels = extract_labels(test_labels_filename, 10000)

# As with our image data, we'll double-check that our 1-hot encoding of the first few values matches our expectations.
print('Training labels shape', train_labels.shape)

# The 1-hot encoding looks reasonable.

# # ### Segmenting data into training, test, and validation
# # The final step in preparing our data is to split it into three sets: training, test, and validation. 
#This isn't the format of the original data set, so we'll take a small slice of the training data and treat 
#that as our validation set.
VALIDATION_SIZE = 5000

validation_data = train_data[:VALIDATION_SIZE, :, :, :]
validation_labels = train_labels[:VALIDATION_SIZE]
train_data = train_data[VALIDATION_SIZE:, :, :, :]
train_labels = train_labels[VALIDATION_SIZE:]

train_size = train_labels.shape[0]
print('Validation shape', validation_data.shape)
print('Train size', train_size)

# check data 
print('\ntrain_data object:', type(train_data), train_data.shape)    
print('train_labels object:', type(train_labels),  train_labels.shape)  
print('validation_data object:', type(validation_data),  validation_data.shape)  
print('validation_labels object:', type(validation_labels),  validation_labels.shape)  
print('test_data object:', type(test_data),  test_data.shape)  
print('test_labels object:', type(test_labels),  test_labels.shape)  

print('\ndata input complete')
# End of code from Google mnist_from_scratch program


# In[13]:


import numpy as np
import pandas as pd
import time

# user-defined function to convert binary digits to digits 0-9
def label_transform(y_in):
    for i in range(len(y_in)):
        if (y_in[i] == 1): return i

y_train = []    
for j in range(train_labels.shape[0]):
    y_train.append(label_transform(train_labels[j,]))  
y_train = np.asarray(y_train)    

y_validation = []    
for j in range(validation_labels.shape[0]):
    y_validation.append(label_transform(validation_labels[j,]))  
y_validation = np.asarray(y_validation)    

y_test = []    
for j in range(test_labels.shape[0]):
    y_test.append(label_transform(test_labels[j,]))  
y_test = np.asarray(y_test)    
    
# 28x28 matrix of entries converted to vector of 784 entries    
X_train = train_data.reshape(55000, 784)
X_validation = validation_data.reshape(5000, 784)    
X_test = test_data.reshape(10000, 784)    

# check data intended for Scikit Learn input
print('\nX_train object:', type(X_train), X_train.shape)    
print('y_train object:', type(y_train),  y_train.shape)  
print('X_validation object:', type(X_validation),  X_validation.shape)  
print('y_validation object:', type(y_validation),  y_validation.shape)  
print('X_test object:', type(X_test),  X_test.shape)  
print('y_test object:', type(y_test),  y_test.shape)      


# In[14]:


# Scikit Learn MLP Classification does validation internally, 
# so there is with no need for a separate validation set.
# We will combine the train and validation sets.

X_train_expanded = np.vstack((X_train, X_validation))
y_train_expanded = np.vstack((y_train.reshape(55000,1), y_validation.reshape(5000,1)))

print('\nX_train_expanded object:', type(X_train_expanded),  X_train_expanded.shape)  
print('y_train_expanded object:', type(y_train_expanded), y_train_expanded.shape)  


# In[7]:


# In[4]

RANDOM_SEED = 9999

from sklearn.neural_network import MLPClassifier

names = ['ANN-2-Layers-10-Nodes-per-Layer',
         'ANN-2-Layers-20-Nodes-per-Layer',
         'ANN-2-Layers-40-Nodes-per-Layer',
         'ANN-5-Layers-10-Nodes-per-Layer',
         'ANN-5-Layers-20-Nodes-per-Layer',
         'ANN-5-Layers-40-Nodes-per-Layer',
         'ANN-10-Layers-10-Nodes-per-Layer',
         'ANN-10-Layers-20-Nodes-per-Layer',
         'ANN-10-Layers-40-Nodes-per-Layer']

layers = [2, 2, 2, 5, 5, 5, 10, 10, 10]
nodes_per_layer = [10, 20, 40, 10, 20, 40, 10, 20, 40]
treatment_condition = [(10, 10), 
                       (20, 20), 
                       (40, 40),
                       (10, 10, 10, 10, 10), 
                       (20, 20, 20, 20, 20),
                       (40, 40, 40, 40, 40),
                       (10, 10, 10, 10, 10, 10, 10, 10, 10, 10), 
                       (20, 20, 20, 20, 20, 20, 20, 20, 20, 20),
                       (40, 40, 40, 40, 40, 40, 40, 40, 40, 40)] 

# note that validation is included in the method  
# for validation_fraction 0.083333, note that 60000 * 0.83333 = 5000    
methods = [MLPClassifier(hidden_layer_sizes=treatment_condition[0], activation='relu', 
              solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, 
              power_t=0.5, max_iter=200, shuffle=True,random_state=RANDOM_SEED, 
              tol=0.0001, verbose=False, warm_start=False, momentum=0.9,nesterovs_momentum=True,  
              early_stopping=False,validation_fraction=0.083333, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
    MLPClassifier(hidden_layer_sizes=treatment_condition[1], activation='relu', 
              solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, 
              power_t=0.5, max_iter=200, shuffle=True, random_state=RANDOM_SEED, 
              tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, 
              early_stopping=False,validation_fraction=0.083333, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
    MLPClassifier(hidden_layer_sizes=treatment_condition[2],activation='relu', 
              solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, 
              power_t=0.5, max_iter=200, shuffle=True, random_state=RANDOM_SEED, 
              tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, 
              early_stopping=False,validation_fraction=0.083333, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
    MLPClassifier(hidden_layer_sizes=treatment_condition[3], activation='relu', 
              solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, 
              power_t=0.5, max_iter=200, shuffle=True, random_state=RANDOM_SEED, 
              tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, 
              early_stopping=False, validation_fraction=0.083333, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
    MLPClassifier(hidden_layer_sizes=treatment_condition[4],activation='relu', 
              solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, 
              power_t=0.5, max_iter=200, shuffle=True, random_state=RANDOM_SEED, 
              tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, 
              early_stopping=False,validation_fraction=0.083333, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
    MLPClassifier(hidden_layer_sizes=treatment_condition[5],activation='relu', 
              solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, 
              power_t=0.5, max_iter=200, shuffle=True, random_state=RANDOM_SEED, 
              tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, 
              early_stopping=False,validation_fraction=0.083333, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
    MLPClassifier(hidden_layer_sizes=treatment_condition[6],activation='relu', 
              solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, 
              power_t=0.5, max_iter=200, shuffle=True, random_state=RANDOM_SEED, 
              tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, 
              early_stopping=False,validation_fraction=0.083333, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
    MLPClassifier(hidden_layer_sizes=treatment_condition[7],activation='relu', 
              solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, 
              power_t=0.5, max_iter=200, shuffle=True, random_state=RANDOM_SEED, 
              tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, 
              early_stopping=False,validation_fraction=0.083333, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
    MLPClassifier(hidden_layer_sizes=treatment_condition[8],activation='relu', 
              solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, 
              power_t=0.5, max_iter=200, shuffle=True, random_state=RANDOM_SEED, 
              tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, 
              early_stopping=False,validation_fraction=0.083333, beta_1=0.9, beta_2=0.999, epsilon=1e-08)]
 
index_for_method = 0 
training_performance_results = []
test_performance_results = []
processing_time = []
   
for name, method in zip(names, methods):
    print('\n------------------------------------')
    print('\nMethod:', name)
    print('\n  Specification of method:', method)
    start_time = time.clock()
    method.fit(X_train, y_train)
    end_time = time.clock()
    runtime = end_time - start_time  # seconds of wall-clock time 
    print("\nProcessing time (seconds): %f" % runtime)        
    processing_time.append(runtime)

    # mean accuracy of prediction in training set
    training_performance = method.score(X_train_expanded, y_train_expanded)
    print("\nTraining set accuracy: %f" % training_performance)
    training_performance_results.append(training_performance)

    # mean accuracy of prediction in test set
    test_performance = method.score(X_test, y_test)
    print("\nTest set accuracy: %f" % test_performance)
    test_performance_results.append(test_performance)
                
    index_for_method += 1

# aggregate the results for final report
# using OrderedDict to preserve the order of variables in DataFrame    
from collections import OrderedDict  

results = pd.DataFrame(OrderedDict([('Method Name', names),
                        ('Layers', layers),
                        ('Nodes per Layer', nodes_per_layer),
                        ('Processing Time', processing_time),
                        ('Training Set Accuracy', training_performance_results),
                        ('Test Set Accuracy', test_performance_results)]))

print('\nBenchmark Experiment: Scikit Learn Artificial Neural Networks\n')
print(results)    

