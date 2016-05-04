# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 16:21:18 2015
Modified on May 3 2016

@author: arash
@editor: junghwanyang
"""
# Import as using different name
import numpy as np
import cPickle as pkl

from time import time

# Use scikit-learn package and load some functions
from sklearn import preprocessing
from sklearn import metrics
# This package has a tree structure.
# Import density under sklearn - utils - extmath
from sklearn.utils.extmath import density
# Similarly, load classifiers from different locations
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.lda import LDA

import json


# Loading training and testing datasets
# This only creates a token to the file location
fp = open('./python_files/training_dataste.pkl', 'rb') # rb: read binary
# Load data into four different objects
[X_train, X_test, Y_train, Y_test] = pkl.load(fp)
# Close after loading
fp.close()

# loading the string of each token or feature or term
fp = open('./python_files/token_str.pkl', 'rb')
feature_names = pkl.load(fp)
fp.close()

# Create a list of categories: C and L. u means UTF (?)
categories = [u'C',u'L']

# Assigining a class label (0, 1) to "Conservatives" or "Liberals"
# It's very similar to recode() in R
le = preprocessing.LabelEncoder()
le.fit(Y_train)
y_train = le.transform(Y_train) 
y_test = le.transform(Y_test) # This I am not very sure of

# Define trim() function for display purpose - only display first 80 characters
def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:80] + "..."
    
    
###############################################################################
# Benchmark classifiers
def benchmark(clf, name): # Read two parameters
    print('_' * 80) # Add one line for presentation
    print("Training: ")
    print(clf)
    t0 = time() # Read the current time
    clf.fit(X_train, y_train) # Training machine ???
    train_time = time() - t0 # Measure the time spent training
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test) # Prediction with test set
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred) # Get accuracy score
    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'): # If the result 'has attribute' coef, print it
        # .shape shows dimensions of a numpy array object        
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))
        # Didn't understand this part 100%
        # The purpose of this is to find top keywords of C and L
        print("top 10 keywords per class:") 
        top10C = np.argsort(clf.coef_[0])[-10:] # Conservative -1
        top10L = np.argsort(clf.coef_[0])[:10] # Liberal +1
        print(trim("C: %s"
                  % (" ".join([feature_names[word_idx] for word_idx in top10C]))))
        print(trim("L: %s"
                  % (" ".join([feature_names[word_idx] for word_idx in top10L]))))

    print("classification report:")
    print(metrics.classification_report(y_test, pred,
                                            target_names=categories))

    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred))

    clf_descr = str(clf).split('(')[0]

    return clf_descr, score, train_time, test_time
    

# Display the accuracy of different classifiers     
results = []
for clf, name in (
        (RidgeClassifier(alpha=1.0,tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (Perceptron(n_iter= 100), "Perceptron"),
        (PassiveAggressiveClassifier(n_iter=100), "Passive-Aggressive"),
        (RandomForestClassifier(n_estimators=10), "Random forest"),
        (LDA(), "Linear Discriminant Analysis"),
        (LinearSVC(), "SVM")
        ):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf, name))   

# Attach classifier to the original json file

# loading dtm file for all twitts
fp = open('./python_files/twitter_dtm.pkl', 'rb')
dtm = pkl.load(fp)
fp.close()

# Predict the labels using Ridges classifier
clf = RidgeClassifier(alpha=1.,tol=1e-2, solver="lsqr")
clf.fit(X_train, y_train)
predicted_labels = clf.predict(dtm)

# loading json file for all twitts
file_name = '../R Project/Data/obamacare.json'
line_reader = open(file_name,'r') # r means for reading

# building a new json file for all twitts + new predicted labels
new_file_name = '../R Project/Data/obamacare_labeled.json'
line_writer = open(new_file_name,'w') # w means for writing

# adding the predicted label to each entry of json file
twit_i = 0
for line in line_reader:
    label = predicted_labels[twit_i]
    if label==0:
        ideology = 'C'
    else:
        ideology = 'L'
    
    line_object = json.loads(line)
    
    
    new_line = '{"ideology":"' + ideology + '",' + line[1:]
    line_writer.write(new_line)
    
    twit_i += 1
line_reader.close()   
line_writer.close() 
