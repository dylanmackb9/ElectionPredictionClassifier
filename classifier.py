import os
import statistics
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import svm
from warnings import simplefilter

# ignoring all future warnings from sklearn
simplefilter(action='ignore', category=FutureWarning)  # ignoring future warnings from sklearn

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



# PREPROCESSING AND FEATURE EXTRACTION

# GRABBING DATA for train_2016
frame = pd.read_csv('train_2016.csv')  # grabbing 2016 full labeled data 
 
fips = pd.DataFrame(frame[['FIPS']]).to_numpy()  # county number
county = pd.DataFrame(frame[['County']]).to_numpy()  # county name 
dem = pd.DataFrame(frame[['DEM']]).to_numpy()  # number of dem votes the county received
gop = pd.DataFrame(frame[['GOP']]).to_numpy()  # number of rep votes the county received 

# taking commas out of median income data and making them ints
medincome = pd.DataFrame(frame[['MedianIncome']]).to_numpy()  # median income of county
for i in range(medincome.shape[0]):
    medincome[i][0] = medincome[i][0].replace(',', "")
medincome = medincome.astype(int)

migrarate = pd.DataFrame(frame[['MigraRate']]).to_numpy()  # migration rate per 1000 (pos if growing, neg if leaving)
brate = pd.DataFrame(frame[['BirthRate']]).to_numpy()  #  frequency of live births per thousand
drate = pd.DataFrame(frame[['DeathRate']]).to_numpy()  # frequency of deaths per thousand
bachrate = pd.DataFrame(frame[['BachelorRate']]).to_numpy()  # percent of people with bachelors degree 
unemployrate = pd.DataFrame(frame[['UnemploymentRate']]).to_numpy()  # percent of labor force unemployed
spread = dem/gop  # gives voting spread, if spread>1 then dem won, spread<1 then gop won
county_winner = (dem > gop)*np.ones((1555,1)) # 1x1555 matrix representing county winner. 1 for dem, 0 for gop

# maximums of all features
dem_max = np.amax(dem)
gop_max = np.amax(gop)
medincome_max = np.amax(medincome)
migrarate_max = np.amax(migrarate)
brate_max = np.amax(brate)
drate_max = np.amax(drate)
bachrate_max = np.amax(bachrate)
unemployrate_max = np.amax(unemployrate)
spread_max = np.amax(spread)

# minimums of all features
dem_min = np.amin(dem)
gop_min = np.amin(gop)
medincome_min = np.amin(medincome)
migrarate_min = np.amin(migrarate)
brate_min = np.amin(brate)
drate_min = np.amin(drate)
bachrate_min = np.amin(bachrate)
unemployrate_min = np.amin(unemployrate)
spread_min = np.amin(spread)

# averages of all features
dem_avg = np.average(dem)
gop_avg = np.average(gop)
medincome_avg = np.average(medincome)
migrarate_avg = np.average(migrarate)
brate_avg = np.average(brate)
drate_avg = np.average(drate)
bachrate_avg = np.average(bachrate)
unemployrate_avg = np.average(unemployrate)

# std of all features

dem_std = np.std(dem)
gop_std = np.std(gop)
medincome_std = np.std(medincome)
migrarate_std = np.std(migrarate)
brate_std = np.std(brate)
drate_std = np.std(drate)
bachrate_std = np.std(bachrate)
unemployrate_std = np.std(unemployrate)

# Z-score standardized features

n_dem = (dem-dem_avg)/dem_std
n_gop = (gop-gop_avg)/gop_std
n_medincome = (medincome-medincome_avg)/medincome_std
n_migrarate = (migrarate-migrarate_avg)/migrarate_std
n_brate = (brate-brate_avg)/brate_std
n_drate = (drate-drate_avg)/drate_std
n_bachrate = (bachrate-bachrate_avg)/bachrate_std
n_unemployrate = (unemployrate-unemployrate_avg)/unemployrate_std


# GRABBING DATA for test_2016_no_label
tframe = pd.read_csv('test_2016_no_label.csv')  # grabbing 2016 submission non-labeled data

t_fips = pd.DataFrame(tframe[['FIPS']]).to_numpy()  # county number
t_county = pd.DataFrame(tframe[['County']]).to_numpy()  # county name 

# taking commas out of median income data and making them ints
t_medincome = pd.DataFrame(tframe[['MedianIncome']]).to_numpy()  # median income of county
for i in range(t_medincome.shape[0]):
    t_medincome[i][0] = t_medincome[i][0].replace(',', "")
t_medincome = t_medincome.astype(int)

t_migrarate = pd.DataFrame(tframe[['MigraRate']]).to_numpy()  # migration rate per 1000 (pos if growing, neg if leaving)
t_brate = pd.DataFrame(tframe[['BirthRate']]).to_numpy()  #  frequency of live births per thousand
t_drate = pd.DataFrame(tframe[['DeathRate']]).to_numpy()  # frequency of deaths per thousand
t_bachrate = pd.DataFrame(tframe[['BachelorRate']]).to_numpy()  # percent of people with bachelors degree 
t_unemployrate = pd.DataFrame(tframe[['UnemploymentRate']]).to_numpy()  # percent of labor force unemployed

# averages of all features
t_medincome_avg = np.average(t_medincome)
t_migrarate_avg = np.average(t_migrarate)
t_brate_avg = np.average(t_brate)
t_drate_avg = np.average(t_drate)
t_bachrate_avg = np.average(t_bachrate)
t_unemployrate_avg = np.average(t_unemployrate)

# std of all features
t_medincome_std = np.std(t_medincome)
t_migrarate_std = np.std(t_migrarate)
t_brate_std = np.std(t_brate)
t_drate_std = np.std(t_drate)
t_bachrate_std = np.std(t_bachrate)
t_unemployrate_std = np.std(t_unemployrate)

# Z-score standardized t_features, based on (xi-avg(x))/std(x)
n_t_medincome = (t_medincome-t_medincome_avg)/t_medincome_std
n_t_migrarate = (t_migrarate-t_migrarate_avg)/t_migrarate_std
n_t_brate = (t_brate-t_brate_avg)/t_brate_std
n_t_drate = (t_drate-t_drate_avg)/t_drate_std
n_t_bachrate = (t_bachrate-t_bachrate_avg)/t_bachrate_std
n_t_unemployrate = (t_unemployrate-t_unemployrate_avg)/t_unemployrate_std





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
            keras.Input(shape=(6,)), 
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

#FEATURE PROCESSING

batch_size = 32

train_dataset = np.dstack((n_medincome, n_migrarate, n_brate, n_drate, n_bachrate, n_unemployrate)).squeeze()  # 1555x6 feature matrix 
labels = county_winner  # 1x1555  labels 

full_xtrain_dataset = train_dataset  # 1555x6 feature matrix
full_ytrain_dataset = county_winner  # 1x1555 labels

k_xtrain_dataset = train_dataset[0:1100].astype('float32')  # Splitting training data for test set
k_ytrain_dataset = np.ravel(labels[0:1100].astype('float32'))  # Splitting training data for test set, 1100,

#combined_dataset = np.hstack((k_xtrain_dataset, k_ytrain_dataset))  # shuffling the data for cross val
#np.random.shuffle(combined_dataset)
#k_xtrain_dataset = combined_dataset[:,0:-1]
#k_ytrain_dataset = combined_dataset[:,-1]

x_test = train_dataset[1100:].astype('float32')  # test set
y_test = labels[1100:].astype('float32')  # test set

tfeatures = np.dstack((n_t_medincome, n_t_migrarate, n_t_brate, n_t_drate, n_t_bachrate, n_t_unemployrate)).squeeze()# SUBMISSION FEATURE MATRIX


# K-FOLD CROSS_VAL for NN

def kcross_nn(k, lr, epo, activ):
    # Implementing k-fold cross validation for neural network 

    # k: number of splits for cross val
    # lr: learning rate 
    # epo: number of training epochs
    # activ: activation function used

    m = int(k_xtrain_dataset.shape[0])  # number of examples
    kaccuracy_list = []
    
    for i in range(k):  # cross val training 
        x_val = k_xtrain_dataset[i*m//k:i*m//k+m//k]  # finding k m/k sized portions
        y_val = k_ytrain_dataset[i*m//k:i*m//k+m//k]  # finding k m/k sized portions
        
        x_train = np.vstack((k_xtrain_dataset[0:i*m//k],k_xtrain_dataset[(i*m//k+m//k):]))  # setting training to be everything but val
        y_train = np.concatenate((k_ytrain_dataset[0:i*m//k],k_ytrain_dataset[(i*m//k+m//k):]))
        
        cNN, self_accuracy = NN(x_train, y_train, epo, activ, lr)  # training nn with created training set above  
 
        pred_val = np.argmax(cNN.predict(x_val), axis=1)  # prediction on validation set
        true_val = y_val  # validation set labels
        
        kaccuracy = weighted_accuracy(pred_val, true_val)  # weighted accuracy for k-fold validation set
        kaccuracy_list.append(kaccuracy)  # appending one of the k accuracies to a list
        
    average_kacc = statistics.mean(kaccuracy_list)  # averaging all k accuracies 
    
    return average_kacc


# K-FOLD CROSS_VAL for SVM

def kcross_svm(k, error, ker):
    # Implementing k-fold cross validation for Support Vector Machine 

    # k: number of splits for cross val
    # error: C value for allowing error
    # ker: kernel used

    m = int(k_xtrain_dataset.shape[0])  # number of training examples
    kaccuracy_list = []

    for i in range(k):  # k-cross training 
        x_val = k_xtrain_dataset[i*m//k:i*m//k+m//k]  # finding k m/k sized portions
        y_val = k_ytrain_dataset[i*m//k:i*m//k+m//k]  # finding k m/k sized portions
        
        x_train = np.vstack((k_xtrain_dataset[0:i*m//k],k_xtrain_dataset[(i*m//k+m//k):]))
        y_train = np.concatenate((k_ytrain_dataset[0:i*m//k],k_ytrain_dataset[(i*m//k+m//k):]))
        
        mod, self_accuracy = SVM(x_train, y_train, error, ker)  # training SVM with training set created above
        
        pred_val = mod.predict(x_val)  # prediction on validation set
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
            accuracy = kcross_svm(k, c, kernel)
            print("For "+kernel+" kernel and "+str(c)+" error value: "+str(accuracy))

def nn_grid():
    # Implementing a Grid Search for nn model tuning on give hyperparameters

    for activ in activation:
        for lr in lr_range:
            for e in epochs:
                accuracy = kcross_nn(k, lr, e, activ)
                print("For "+activ+" activation and "+str(lr)+" learning rate and "+str(e)+" epochs: "+str(accuracy))
                

#svm_grid()  # running grid search


# TRAINING

# ideal svm
# best: 

#optSVM = svm.SVC(C=10, kernel='rbf')
#optSVM.fit(full_xtrain_dataset,full_ytrain_dataset)
#preds_svm = optSVM.predict(tfeatures)



#ideal nn
# best: relu .01, 20 
     #: relu .1 10
     #: sig .01 20 

optNN, accuracy = NN(full_xtrain_dataset, full_ytrain_dataset, 9, 'relu', .1 )  # training optimized NN on full dataset  



# EVALUATING

pred_test = np.argmax(optNN.predict(x_test), axis=1)  # predictions on test set 
true_test = y_test  # test set labels
accuracy_test = weighted_accuracy(pred_test, true_test)  # test set weighted accuracy

print("Accuracy on test set: "+str(accuracy_test))
print("")
pred_sub = np.argmax(optNN.predict(tfeatures), axis=1)  #predictions submission set
print("Number of counties predicted dem on test set: "+str(np.sum(pred_test)))
print("Number of counties predicted dem on submission set: "+str(np.sum(pred_sub)))
print("")
print("Test set dem/total ratio: "+str(np.sum(pred_test)/(1555-1100)))  # 1555-1100 is size of test set
print("Submission set dem/total ratio: "+str(np.sum(pred_sub)/1555))  # 1555 size of submission set





graphdata = pd.read_csv('graph.csv').to_numpy()  # grabbing graph data
#print(type(graphdata))

# GRABBING DATA for train_2012
cframe = pd.read_csv('train_2012.csv')  # grabbing 2016 full labeled data 
 
cfips = pd.DataFrame(cframe[['FIPS']]).to_numpy()  # county number
ccounty = pd.DataFrame(cframe[['County']]).to_numpy()  # county name 
cdem = pd.DataFrame(cframe[['DEM']]).to_numpy()  # number of dem votes the county received
cgop = pd.DataFrame(cframe[['GOP']]).to_numpy()  # number of rep votes the county received 

# taking commas out of median income data and making them ints
cmedincome = pd.DataFrame(cframe[['MedianIncome']]).to_numpy()  # median income of county
for i in range(cmedincome.shape[0]):
    cmedincome[i][0] = cmedincome[i][0].replace(',', "")
cmedincome = cmedincome.astype(int)

cmigrarate = pd.DataFrame(cframe[['MigraRate']]).to_numpy()  # migration rate per 1000 (pos if growing, neg if leaving)
cbrate = pd.DataFrame(cframe[['BirthRate']]).to_numpy()  #  frequency of live births per thousand
cdrate = pd.DataFrame(cframe[['DeathRate']]).to_numpy()  # frequency of deaths per thousand
cgrowthrate = cbrate-cdrate  # NEW FEATURE showing rate at which county population grows
cbachrate = pd.DataFrame(cframe[['BachelorRate']]).to_numpy()  # percent of people with bachelors degree 
cunemployrate = pd.DataFrame(cframe[['UnemploymentRate']]).to_numpy()  # percent of labor force unemployed
cspread = cdem/cgop  # gives voting spread, if spread>1 then dem won, spread<1 then gop won
ccounty_winner = (cdem > cgop)*np.ones((1555,1)) # 1x1555 matrix representing county winner. 1 for dem, 0 for gop

# averages of all features
cdem_avg = np.average(cdem)
cgop_avg = np.average(cgop)
cmedincome_avg = np.average(cmedincome)
cmigrarate_avg = np.average(cmigrarate)
cbrate_avg = np.average(cbrate)
cdrate_avg = np.average(cdrate)
cgrowthrate_avg = np.average(cgrowthrate)
cbachrate_avg = np.average(cbachrate)
cunemployrate_avg = np.average(cunemployrate)

# std of all features

cdem_std = np.std(cdem)
cgop_std = np.std(cgop)
cmedincome_std = np.std(cmedincome)
cmigrarate_std = np.std(cmigrarate)
cbrate_std = np.std(cbrate)
cdrate_std = np.std(cdrate)
cgrowthrate_std = np.std(cgrowthrate)
cbachrate_std = np.std(cbachrate)
cunemployrate_std = np.std(cunemployrate)

# Z-score standardized features

n_cdem = (cdem-cdem_avg)/cdem_std
n_cgop = (cgop-cgop_avg)/cgop_std
n_cmedincome = (cmedincome-cmedincome_avg)/cmedincome_std
n_cmigrarate = (cmigrarate-cmigrarate_avg)/cmigrarate_std
n_cbrate = (cbrate-cbrate_avg)/cbrate_std
n_cdrate = (cdrate-cdrate_avg)/cdrate_std
n_cgrowthrate = (cgrowthrate-cgrowthrate_avg)/cgrowthrate_std
n_cbachrate = (cbachrate-cbachrate_avg)/cbachrate_std
n_cunemployrate = (cunemployrate-cunemployrate_avg)/cunemployrate_std


# GRABBING DATA for test_2012_no_label
tcframe = pd.read_csv('test_2012_no_label.csv')  # grabbing 2012 submission non-labeled data

t_cfips = pd.DataFrame(tcframe[['FIPS']]).to_numpy()  # county number
t_ccounty = pd.DataFrame(tcframe[['County']]).to_numpy()  # county name 

# taking commas out of median income data and making them ints
t_cmedincome = pd.DataFrame(tcframe[['MedianIncome']]).to_numpy()  # median income of county
for i in range(t_cmedincome.shape[0]):
    t_cmedincome[i][0] = t_cmedincome[i][0].replace(',', "")
t_cmedincome = t_cmedincome.astype(int)

t_cmigrarate = pd.DataFrame(tcframe[['MigraRate']]).to_numpy()  # migration rate per 1000 (pos if growing, neg if leaving)
t_cbrate = pd.DataFrame(tcframe[['BirthRate']]).to_numpy()  #  frequency of live births per thousand
t_cdrate = pd.DataFrame(tcframe[['DeathRate']]).to_numpy()  # frequency of deaths per thousand
t_cgrowthrate = t_cbrate-t_cdrate
t_cbachrate = pd.DataFrame(tcframe[['BachelorRate']]).to_numpy()  # percent of people with bachelors degree 
t_cunemployrate = pd.DataFrame(tcframe[['UnemploymentRate']]).to_numpy()  # percent of labor force unemployed

# averages of all features
t_cmedincome_avg = np.average(t_cmedincome)
t_cmigrarate_avg = np.average(t_cmigrarate)
t_cbrate_avg = np.average(t_cbrate)
t_cdrate_avg = np.average(t_cdrate)
t_cgrowthrate_avg = np.average(t_cgrowthrate)
t_cbachrate_avg = np.average(t_cbachrate)
t_cunemployrate_avg = np.average(t_cunemployrate)

# std of all features
t_cmedincome_std = np.std(t_cmedincome)
t_cmigrarate_std = np.std(t_cmigrarate)
t_cbrate_std = np.std(t_cbrate)
t_cdrate_std = np.std(t_cdrate)
t_cgrowthrate_std = np.std(t_cgrowthrate)
t_cbachrate_std = np.std(t_cbachrate)
t_cunemployrate_std = np.std(t_cunemployrate)

# Z-score standardized t_features, based on (xi-avg(x))/std(x)
n_t_cmedincome = (t_cmedincome-t_cmedincome_avg)/t_cmedincome_std
n_t_cmigrarate = (t_cmigrarate-t_cmigrarate_avg)/t_cmigrarate_std
n_t_cbrate = (t_cbrate-t_cbrate_avg)/t_cbrate_std
n_t_cdrate = (t_cdrate-t_cdrate_avg)/t_cdrate_std
n_t_cgrowthrate = (t_cgrowthrate-t_cgrowthrate_avg)/t_cgrowthrate_std
n_t_cbachrate = (t_cbachrate-t_cbachrate_avg)/t_cbachrate_std
n_t_cunemployrate = (t_cunemployrate-t_cunemployrate_avg)/t_cunemployrate_std




#INITIAL PROCESSING

#2016

train_dataset_features2016 = np.dstack((n_medincome, n_migrarate, n_brate, n_drate, n_growthrate, n_bachrate, n_unemployrate)).squeeze()  # 1555x7 feature matrix 
train_dataset_labels2016 = county_winner  # 1x1555  labels 

full_xtrain_dataset = train_dataset_features2016  # 1555x6 feature matrix
full_ytrain_dataset = train_dataset_labels2016  # 1x1555 labels

k_xtrain_dataset = train_dataset_features2016[0:1100].astype('float32')  # Splitting training data for test set
k_ytrain_dataset = np.ravel(train_dataset_labels2016[0:1100].astype('float32'))  # Splitting training data for test set

#combined_dataset = np.hstack((k_xtrain_dataset, k_ytrain_dataset))  # shuffling the data for cross val
#np.random.shuffle(combined_dataset)
#k_xtrain_dataset = combined_dataset[:,0:-1]
#k_ytrain_dataset = combined_dataset[:,-1]

x_test = train_dataset_features2016[1100:].astype('float32')  # test set
y_test = train_dataset_labels2016[1100:].astype('float32')  # test set

tfeatures2016 = np.dstack((n_t_medincome, n_t_migrarate, n_t_brate, n_t_drate, n_t_bachrate, n_t_unemployrate)).squeeze()# SUBMISSION FEATURE MATRIX

full_train_dataset2016 = np.hstack((train_dataset_features2016, train_dataset_labels2016))  # combined data


#2012

train_dataset_features2012 = np.dstack((n_cmedincome, n_cmigrarate, n_cbrate, n_cdrate, n_cgrowthrate, n_cbachrate, n_cunemployrate)).squeeze()  # 1555x6 feature matrix 
train_dataset_labels2012 = ccounty_winner  # 1x1555  labels 

cfull_xtrain_dataset = train_dataset_features2012  # 1555x6 feature matrix
cfull_ytrain_dataset = train_dataset_labels2012  # 1x1555 labels

ck_xtrain_dataset = train_dataset_features2012[0:1100].astype('float32')  # Splitting training data for test set
ck_ytrain_dataset = np.ravel(train_dataset_labels2012[0:1100].astype('float32'))  # Splitting training data for test set

#combined_dataset = np.hstack((k_xtrain_dataset, k_ytrain_dataset))  # shuffling the data for cross val
#np.random.shuffle(combined_dataset)
#k_xtrain_dataset = combined_dataset[:,0:-1]
#k_ytrain_dataset = combined_dataset[:,-1]

cx_test = train_dataset_features2012[1100:].astype('float32')  # test set
cy_test = train_dataset_labels2012[1100:].astype('float32')  # test set

tfeatures2012 = np.dstack((n_t_cmedincome, n_t_cmigrarate, n_t_cbrate, n_t_cdrate, n_t_cgrowthrate, n_t_cbachrate, n_t_cunemployrate)).squeeze()# SUBMISSION FEATURE MATRIX

full_train_dataset2012 = np.hstack((train_dataset_features2012, train_dataset_labels2012))  # combined data



# KNN
def get_kNN(fip_county):
  # Finding all counties (as fip) that fip_county connects to
  
  #fip_county: county number as int that
  nearest_fips=[]
  nearest_indices = np.where(graphdata[:,0]==fip_county)[0]  # array of indices
  for i in range(nearest_indices.shape[0]):
    if graphdata[:,-1][i] != fip_county:
      nearest_fips.append(graphdata[:,-1][i])


  return nearest_fips



# KNN on 2016 data using 2012 data as training 

#Creating new feature for 2016 using kNN on 2012 data

geoSwing = np.zeros((medincome.shape[0],))  # new feature array of size 1555,
for i in range(medincome.shape[0]):  # iterating over all examples 
  current_fip = fips[i]  # iterating over each fip in 2016 data
  nearest_list = get_kNN(current_fip)  # grabbing list of connected fips
  swing_var = 0  # if the county is connected to gop or dem counties
  for j in nearest_list:  # going through list of closest counties
    if (np.where(cfips==j)[0]).size > 0:  # seeing if label exists
      if (cfull_ytrain_dataset[np.where(cfips==j)[0][0]]) == 0:  # seeing if its labeled gop
        swing_var = swing_var - 1  # more influence towards gop
      elif (cfull_ytrain_dataset[np.where(cfips==j)[0][0]]) == 1:  # seeing if labeled dem
        swing_var = swing_var + 1  # more influence towards dem

  geoSwing[i] = swing_var  # new Feature


geoSwing = np.reshape(geoSwing, ((1555,1)))


#FINAL PROCESSING

train_dataset_features2016 = np.dstack((n_medincome, n_migrarate, n_brate, n_drate, n_growthrate, n_bachrate, n_unemployrate, geoSwing)).squeeze()  # 1555x8 feature matrix
train_dataset_labels2016 = county_winner  # labels 

tfeatures2016 = np.dstack((n_t_medincome, n_t_migrarate, n_t_brate, n_t_drate, n_t_growthrate, n_t_bachrate, n_t_unemployrate, geo_pred)).squeeze()# SUBMISSION FEATURE MATRIX

full_train_dataset2016 = np.hstack((train_dataset_features2016, train_dataset_labels2016))

k_xtrain_dataset = train_dataset_features2016[0:1100].astype('float32')  # Splitting training data for test set
k_ytrain_dataset = np.ravel(train_dataset_labels2016[0:1100].astype('float32'))  # Splitting training data for test set

x_test = train_dataset_features2016[1100:].astype('float32')  # test set
y_test = train_dataset_labels2016[1100:].astype('float32')  # test set



#testing

print(train_dataset_features2016.shape)  # 2016 features
print(train_dataset_labels2016.shape)  # 2016 labels
print(tfeatures2016.shape)  # 2016 test features
print("")
print(cfull_xtrain_dataset.shape)  # 2012 features
print(cfull_ytrain_dataset.shape)  # 2012 labels
print(tfeatures2012.shape)  # 2012 test features



# CREATING SUBSAMPLE: even number of positively and negatively labeled examples

posclass_indices = np.where(full_train_dataset2016[:,-1] == 1)  # finding all indices of positively labeled examples
negclass_indices = np.where(full_train_dataset2016[:,-1] == 0)  # finding all indices of negatively labeled examples
num_features = full_train_dataset2016.shape[-1]  # number of features


posclass_arr = np.zeros((1,num_features))  # initiating positive class array, 1xfeatures
for i in range(posclass_indices[0].shape[0]):  # creating array of all positively labeled examples
  posclass_arr = np.concatenate((posclass_arr, np.reshape(full_train_dataset2016[posclass_indices[0][i]], (1,num_features))), axis=0)

posclass_arr = posclass_arr[1:,:]  # removing initiated zero vector 


negclass_arr = np.zeros((1,num_features))  # initiating negative class array, 1xfeatures
for i in range(posclass_indices[0].shape[0]):  # creating array of all negatively labeled examples
  negclass_arr = np.concatenate((negclass_arr, np.reshape(full_train_dataset2016[negclass_indices[0][i]], (1,num_features))))

negclass_arr = negclass_arr[1:,:]  # removing initiated zero vector

subsample = np.concatenate((posclass_arr, negclass_arr), axis=0)  # creating 450x7 subsample of 225 pos and 225 neg examples
np.random.shuffle(subsample)  # shuffling

subsample_features = subsample[:,0:-1]  # subsample feature matrix 
subsample_labels = subsample[:,-1]  # subsample labels



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


# creating geoSwing for test_2016 by predicting geoSwing for test_2012

geotrain_feature = cfull_xtrain_dataset  # 2012 features without geoSwing 
geotrain_label = geoSwing+3  #  2012 labels, adding 3 so multiclass neural net predicts from 0-4

gNN, accuracy = geoNN(geotrain_feature, geotrain_label, 80, 'relu', .00001)  # training a NN on 2012 data with geoSwing labels
geo_pred = np.argmax(gNN.predict(tfeatures2012), axis=1)  # predictions on 2012 test data


# PREDICTING ON SUBMISSION DATA
gsubNN, accuracy = NN(train_dataset_features2016, train_dataset_labels2016, 50, 'relu', .01 )  # training optimized NN on full entire dataset  
pred_sub = np.argmax(gsubNN.predict(tfeatures2016), axis=1)  #predictions submission set
print("Number of counties predicted dem on submission set: "+str(np.sum(pred_sub)))
print("Submission set dem/total ratio: "+str(np.sum(pred_sub)/1555))  # 1555 size of submission set



#Creating submission file 


#BASIC
solution = np.stack((np.ravel(t_fips.astype(int)), pred_sub.astype(int)), axis=-1).squeeze()  # combining county number and predictions
header = np.array(['FIPS','Result'])  # header for submission file
export_df = pd.DataFrame(solution, columns=header)
export_df.to_csv('predictions_f_11', header=True, index=False)







