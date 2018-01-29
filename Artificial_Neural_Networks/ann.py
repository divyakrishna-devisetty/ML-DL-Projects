# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
#independent variable/feature vector
X = dataset.iloc[:, 3:13].values
# dependent variable vector
y = dataset.iloc[:, 13].values

# Encoding categorical data ; dependent is already encoded to 0 and 1
# we have two categorical independent variables, country and gender
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
# to create dummy variables for country as it has three categories
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
# Dummy variable trap if spain and germany is 0 that means france 
# is ought to be 1, so droping dummy fr france doesnt make any impact
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - the ANN!

# Importing the Keras libraries and packages
import keras
# To initialize our ANN
from keras.models import Sequential

# To build hidden layers of ANN
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
'''
output_dim --> no of nodes in hidden layer i.e., 6(Tip: avg of input nodes and output nodes)
input nodes --> features /independent variable
output nodes --> output is categorical variable with binary outcome problem so only one node
Another tip:deciding hidden nodes using k-fold or cross validation modelling
where in we can experiment with number of hidden nodes etc..
other than tran n test set we keep aside this cross validation data
init--> uniformly intialise weights close to 0
activation--> we choose rectifier activation function(phi(x)=max(x,0))
for hidden layyer(relu) and sigmoid fr output layer
input_dim -->no. of independent variables

'''
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
'''
We are making geodemographic segmanation model we expect probabilites as outcome
probabiltiy that each customer leaves bank so we use sigmoid fn.
here we have two categories leaves bank or stays in bank.
if dependent variable has more than three categories then we use softmax fn.
'''
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
'''
adam--one of the stochastic gradient descent algorithm
binary_crossentropy - logarithmic loss fn for two category dependent varible
categorical_crossentropy = more than 2 category output variable
'''
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
#epoch - no.of times we train our ANN on whole training set, accuracy imporves over the num of epochs
#batch size after which we choose to update weights
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
'''
Epoch 100/100
8000/8000 [==============================] - 1s 124us/step - loss: 0.3998 - acc: 0.8359
'''
# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
# analysing ouput: 1st customer of test set at row 0 has 20% prob to leave bank
y_pred = classifier.predict(X_test)
# Threshold is chosen randomly; prob >0.5 wll leave bank evaluating to True and viceversa
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# accuracy = no.of correct predictions/total_predictions
# 1537+150/2000 = 0.8435
