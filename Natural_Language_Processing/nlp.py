# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 13:35:13 2018

@author: Divya krishna
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter = '\t', quoting = 3)

# cleaning the test
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = [] # corpus is collections of sentences of same type
for i in range(1000):
    review = re.sub(r'[^a-zA-Z]',' ',dataset['Review'][i]) #remove all nums n non alpha numerics
    review = review.lower()
    # remove irrelavant words which dont help in tagging reviews good or bad
    # such as articles prepositions etc
    
    review = review.split() # to convert string to list, to iterate and remove stopwords
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    # stem the words to avoid different forms of same word in sparse matrix
    # cleaned review
    review = ' '.join(review)
    corpus.append(review)

# creating bag of words model(unique words)/ sparse matrix
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500) # without max_features you get 1565 columns/unique wrds
#by adding max_features we remove irrelavant words.keep upto 1500 unique wrds.
# sparse matrix with columns as features/independent variables
X = cv.fit_transform(corpus).toarray()
# include dependent variable
y = dataset.iloc[:,1].values

# splitting the data set into the training set and test set
from sklearn.cross_validation import train_test_split
# random state=0 coz we want to expect all the same results ireespective of runs.
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)


# fitting naive bayes to training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# predicting the test set results
y_pred = classifier.predict(X_test)

#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# out of 200 reviews of test set NB model predicted 55 negative reviews crctly
# 91 positive reviews coreectly and mis predicted 42 negative reviews as positive
# 12 positive reviews as negative. More the data lesser the misprediction.
accuracy = 55+91/200
