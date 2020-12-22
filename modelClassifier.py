import os
import statistics
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import svm
from warnings import simplefilter

import dataProcessor as data

# ignoring all future warnings from sklearn
simplefilter(action='ignore', category=FutureWarning)  # ignoring future warnings from sklearn
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


#pd.__version__
#print(tf.__version__)
#print(tf.executing_eagerly())


def weighted_accuracy(pred, true):
	# Function for our accuracy metric 
    # pred: 1555x1 
    # true: 1555x1 
    
    assert(len(pred) == len(true))
    num_labels = len(true)
    num_pos = sum(true)  # number of dem
    num_neg = num_labels - num_pos  # number of gop
    frac_pos = num_pos/num_labels  # % labeled dems
    weight_pos = 1/frac_pos
    weight_neg = 1/(1-frac_pos)
    num_pos_correct = 0
    num_neg_correct = 0
    for pred_i, true_i in zip(pred, true):
        num_pos_correct += (pred_i == true_i and true_i == 1)
        num_neg_correct += (pred_i == true_i and true_i == 0)
    weighted_accuracy = ((weight_pos * num_pos_correct) 
                         + (weight_neg * num_neg_correct))/((weight_pos * num_pos) + (weight_neg * num_neg))
    return weighted_accuracy


#TRAINING ALGORITHMS

#DENSE NEURAL NET
def NN(x_train, y_train, epo, activ, lr):
    # Creating a fully connected sequential neural network 
    # Should be using eager execution
    
    # x_train: training set, Ax6 ndarray
    # y_train: training labels  1xB ndarray
    # epo: Number of training epochs it will run (int)
    # activ: activation function, 'relu','sigmoid','linear' (string)
    # lr: learning rate (int) 
 

    myNN = keras.Sequential(
        [
            keras.Input(shape=(7,)), 
            layers.Dense(10, activation=activ, name='layer1'),  # 2 hidden layers
            layers.Dense(10, activation=activ, name='layer2'),
            layers.Dense(2, activation='softmax'),  # softmax on output
        ]
    )
    
    myNN.compile(
        optimizer = keras.optimizers.Adam(learning_rate=lr),  # Using Adam (SGD) as iterative optimizer
        loss = keras.losses.SparseCategoricalCrossentropy(),  # Using SparseCategoricalCrossEntropy as loss
        metrics = ['accuracy'],
    )
        
    myNN.fit(x_train, y_train, epochs=epo, verbose=0)  # fitting 
    #print("")
    #print("")
    
    preds = np.argmax(myNN.predict(x_train), axis=1)  # predicting on training set
    true = y_train  # training set labels 
    accuracy = weighted_accuracy(preds,true)  # weighted accuracy on training
    
    
    return myNN, accuracy 



def SVM(x_train, y_train, error_c, ker):
    # Creating a kernelized soft-margin Support Vector Machine
    
    #x_train: training set, Ax6 ndarray
    #y_train: training labels, 1xB ndarray
    #error_c:  error constant, as C->inf, SVM becomes hard margin
    #ker: kernel used 
    
    clf = svm.SVC(C=error_c, kernel=ker)  # model 
    clf.fit(x_train,y_train)  # fitting on training data
    preds = clf.predict(x_train)  # predicting on training set
    true = y_train  # training set labels
    accuracy = weighted_accuracy(preds,true)  # weighted accuracy on training
    return clf, accuracy
    


# TRAINING, VALIDATION, MODEL SELECTION


# K-FOLD CROSS_VAL for NN

def kcross_nn(k_xtrain, k_ytrain, k, lr, epo, activ):
    # Implementing k-fold cross validation for neural network 

    # k: number of splits for cross val
    # lr: learning rate 
    # epo: number of training epochs
    # activ: activation function used

    m = int(k_xtrain.shape[0])  # number of examples
    kaccuracy_list = []
    
    for i in range(k):  # cross val training 
        x_val = k_xtrain[i*m//k:i*m//k+m//k]  # finding k m/k sized portions
        y_val = k_ytrain[i*m//k:i*m//k+m//k]  # finding k m/k sized portions
        
        x_train = np.vstack((k_xtrain[0:i*m//k],k_xtrain[(i*m//k+m//k):]))  # setting training to be everything but val
        y_train = np.concatenate((k_ytrain[0:i*m//k],k_ytrain[(i*m//k+m//k):]))
        
        cNN, self_accuracy = NN(x_train, y_train, epo, activ, lr)  # training nn with created training set above  
 
        pred_val = np.argmax(cNN.predict(x_val), axis=1)  # prediction on validation set
        true_val = y_val  # validation set labels
        
        kaccuracy = weighted_accuracy(pred_val, true_val)  # weighted accuracy for k-fold validation set
        kaccuracy_list.append(kaccuracy)  # appending one of the k accuracies to a list
        
    average_kacc = statistics.mean(kaccuracy_list)  # averaging all k accuracies 
    
    return average_kacc


# K-FOLD CROSS_VAL for SVM

def kcross_svm(k_xtrain, k_ytrain, k, error, ker):
    # Implementing k-fold cross validation for Support Vector Machine 

    # k: number of splits for cross val
    # error: C value for allowing error
    # ker: kernel used

    m = int(k_xtrain.shape[0])  # number of training examples
    kaccuracy_list = []

    for i in range(k):  # k-cross training 
        x_val = k_xtrain[i*m//k:i*m//k+m//k]  # finding k m/k sized portions
        y_val = k_ytrain[i*m//k:i*m//k+m//k]  # finding k m/k sized portions
        
        x_train = np.vstack((k_xtrain[0:i*m//k],k_xtrain[(i*m//k+m//k):]))
        y_train = np.concatenate((k_ytrain[0:i*m//k],k_ytrain[(i*m//k+m//k):]))
        
        Smodel, self_accuracy = SVM(x_train, y_train, error, ker)  # training SVM with training set created above
        
        pred_val = Smodel.predict(x_val)  # prediction on validation set
        true_val = y_val  # validation set labels
        
        kaccuracy = weighted_accuracy(pred_val,true_val)  # finding weighted accuracy on validation set
        kaccuracy_list.append(kaccuracy)  # appending one of the k accuracies to list
        
    average_acc = statistics.mean(kaccuracy_list)  # averaging k accuracies  
    
    return average_acc
        

#MODEL TUNING

#Hyperparameters for SVM
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
error_range =[.001, .01, .1, 1, 10]   

#Hyperparameters for NN
lr_range = [.0001,.001, .01, .1, 1]
epochs = [2, 5, 10, 20, 30, 50]
activation = ['linear','relu','sigmoid']

k = 10 # based on 1100 size training set

def svm_grid():
    # Implementing a Grid Search for svm model tuning on give hyperparameters

    for kernel in kernels:
        for c in error_range:
            accuracy = kcross_svm(data.k_xtrain_dataset, data.k_ytrain_dataset, k,  c, kernel)
            print("For "+kernel+" kernel and "+str(c)+" error value: "+str(accuracy))

def nn_grid():
    # Implementing a Grid Search for nn model tuning on give hyperparameters

    for activ in activation:
        for lr in lr_range:
            for e in epochs:
                accuracy = kcross_nn(data.k_xtrain_dataset, data.k_ytrain_dataset, k, lr, e, activ)
                print("For "+activ+" activation and "+str(lr)+" learning rate and "+str(e)+" epochs: "+str(accuracy))
                

#svm_grid()  # running grid search



# TRAINING


#optSVM = svm.SVC(C=10, kernel='rbf')
#optSVM.fit(full_xtrain_dataset,full_ytrain_dataset)
#preds_svm = optSVM.predict(tfeatures)



#ideal nn
# best: relu .01, 20 
     #: relu .1 10
     #: sig .01 20 

#print(data.full_xtrain_dataset.shape)
#print(data.full_ytrain_dataset.shape)

optNN, accuracy = NN(data.full_xtrain_dataset, data.full_ytrain_dataset, 9, 'relu', .1 )  # training optimized NN on full dataset  



# EVALUATING

pred_test = np.argmax(optNN.predict(data.x_test), axis=1)  # predictions on test set 
true_test = data.y_test  # test set labels
accuracy_test = weighted_accuracy(pred_test, true_test)  # test set weighted accuracy

print("Accuracy on test set: "+str(accuracy_test))
print("")
pred_sub = np.argmax(optNN.predict(data.tfeatures2016), axis=1)  #predictions submission set
print("Number of counties predicted dem on test set: "+str(np.sum(pred_test)))
print("Number of counties predicted dem on submission set: "+str(np.sum(pred_sub)))
print("")
print("Test set dem/total ratio: "+str(np.sum(pred_test)/(1555-1100)))  # 1555-1100 is size of test set
print("Submission set dem/total ratio: "+str(np.sum(pred_sub)/1555))  # 1555 size of submission set






#DENSE NEURAL NET FOR GEOSWING PREDICTION
def geoNN(x_train, y_train, epo, activ, lr):
    # Creating a fully connected sequential neural network 
    # Should be using eager execution
    # multiclass from 0-4
    
    # x_train: training set, Ax7 ndarray
    # y_train: training labels  1xB ndarray
    # epo: Number of training epochs it will run (int)
    # activ: activation function, 'relu','sigmoid','linear' (string)
    # lr: learning rate (int) 
 

    myNN = keras.Sequential(
        [
            keras.Input(shape=(7,)), 
            layers.Dense(10, activation=activ, name='layer1'),  # 2 hidden layers
            layers.Dense(10, activation=activ, name='layer2'),
            layers.Dense(5, activation='softmax'),  # softmax on output
        ]
    )
    
    myNN.compile(
        optimizer = keras.optimizers.Adam(learning_rate=lr),  # Using Adam (SGD) as iterative optimizer
        loss = keras.losses.SparseCategoricalCrossentropy(),  # Using SparseCategoricalCrossEntropy as loss
        metrics = ['accuracy'],
    )
        
    myNN.fit(x_train, y_train, epochs=epo, verbose=0)  # fitting 
    #print("")
    #print("")
    
    preds = np.argmax(myNN.predict(x_train), axis=1)  # predicting on training set
    true = y_train  # training set labels 
    accuracy = weighted_accuracy(preds,true)  # weighted accuracy on training
    
    
    return myNN, accuracy 



# KNN
def get_kNN(fip_county):
  # Finding all counties (as fip) that fip_county connects to
  
  #fip_county: county number as int that
  nearest_fips=[]
  nearest_indices = np.where(data.graphdata[:,0]==fip_county)[0]  # array of indices
  for i in range(nearest_indices.shape[0]):
    if data.graphdata[:,-1][i] != fip_county:
      nearest_fips.append(data.graphdata[:,-1][i])

  return nearest_fips



# KNN on 2016 data using 2012 data as training 

#Creating new feature for 2016 using kNN on 2012 data

geoSwing = np.zeros((data.medincome.shape[0],))  # new feature array of size 1555,
for i in range(data.medincome.shape[0]):  # iterating over all examples 
  current_fip = data.fips[i]  # iterating over each fip in 2016 data
  nearest_list = get_kNN(current_fip)  # grabbing list of connected fips
  swing_var = 0  # if the county is connected to gop or dem counties
  for j in nearest_list:  # going through list of closest counties
    if (np.where(data.cfips==j)[0]).size > 0:  # seeing if label exists
      if (data.cfull_ytrain_dataset[np.where(data.cfips==j)[0][0]]) == 0:  # seeing if its labeled gop
        swing_var = swing_var - 1  # more influence towards gop
      elif (data.cfull_ytrain_dataset[np.where(data.cfips==j)[0][0]]) == 1:  # seeing if labeled dem
        swing_var = swing_var + 1  # more influence towards dem

  geoSwing[i] = swing_var  # new Feature

geoSwing = np.reshape(geoSwing, ((1555,1)))  # GeoSwing for 2016 training features

geotrain_feature = data.cfull_xtrain_dataset  # 2012 features without geoSwing 
geotrain_label = geoSwing+3  #  2012 labels, adding 3 so multiclass neural net predicts from 0-4

gNN, accuracy = geoNN(geotrain_feature, geotrain_label, 80, 'relu', .00001)  # training a NN on 2012 data with geoSwing labels
geo_pred = np.argmax(gNN.predict(data.tfeatures2012), axis=1).reshape((1555,1))  # predictions on 2012 test data


# FINAL PROCESSING


train_dataset_features2016_add = np.hstack((data.train_dataset_features2016, geoSwing))  # training data set with new feature
train_dataset_labels2016_add = data.county_winner  # labels 

tfeatures2016_add = np.hstack((data.tfeatures2016, geo_pred)) # SUBMISSION FEATURE MATRIX

full_train_dataset2016_add = np.hstack((train_dataset_features2016_add, train_dataset_labels2016_add))

k_xtrain_dataset_add = train_dataset_features2016_add[0:1100].astype('float32')  # Splitting training data for test set
k_ytrain_dataset_add = np.ravel(train_dataset_labels2016_add[0:1100].astype('float32'))  # Splitting training data for test set

x_test_add = train_dataset_features2016_add[1100:].astype('float32')  # test set
y_test_add = train_dataset_labels2016_add[1100:].astype('float32')  # test set

#testing

print(data.train_dataset_features2016.shape)  # 2016 features
print(data.train_dataset_labels2016.shape)  # 2016 labels
print(data.tfeatures2016.shape)  # 2016 test features
print("")
print(data.cfull_xtrain_dataset.shape)  # 2012 features
print(data.cfull_ytrain_dataset.shape)  # 2012 labels
print(data.tfeatures2012.shape)  # 2012 test features
print("")
print(train_dataset_features2016_add.shape)  # 2016 features after adding GeoSwing feature
print(train_dataset_labels2016_add.shape)  # 2016 labels after adding GeoSwing feature
print(tfeatures2016_add.shape)  # 2016 submission features after adding GeoSwing feature



# PREDICTING ON SUBMISSION DATA
gsubNN, accuracy = NN(data.train_dataset_features2016, data.train_dataset_labels2016, 50, 'relu', .01 )  # training optimized NN on full entire dataset  
pred_sub = np.argmax(gsubNN.predict(data.tfeatures2016), axis=1)  #predictions submission set
print("Number of counties predicted dem on submission set: "+str(np.sum(pred_sub)))
print("Submission set dem/total ratio: "+str(np.sum(pred_sub)/1555))  # 1555 size of submission set


#Creating submission file 


#BASIC
solution = np.stack((np.ravel(data.t_fips.astype(int)), pred_sub.astype(int)), axis=-1).squeeze()  # combining county number and predictions
header = np.array(['FIPS','Result'])  # header for submission file
export_df = pd.DataFrame(solution, columns=header)
#export_df.to_csv('predictions_f_11', header=True, index=False)







