# PREPROCESSING AND FEATURE EXTRACTION

# GRABBING DATA for train_2016


import pandas as pd
import numpy as np


graphdata = pd.read_csv('graph.csv').to_numpy()  # grabbing graph data
#print(type(graphdata))


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
growthrate = brate-drate  # NEW FEATURE showing rate at which county population grows
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
growthrate_max = np.amax(growthrate)
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
growthrate_min = np.amin(growthrate)
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
growthrate_avg = np.average(growthrate)
bachrate_avg = np.average(bachrate)
unemployrate_avg = np.average(unemployrate)

# std of all features

dem_std = np.std(dem)
gop_std = np.std(gop)
medincome_std = np.std(medincome)
migrarate_std = np.std(migrarate)
brate_std = np.std(brate)
drate_std = np.std(drate)
growthrate_std = np.std(growthrate)
bachrate_std = np.std(bachrate)
unemployrate_std = np.std(unemployrate)

# Z-score standardized features

n_dem = (dem-dem_avg)/dem_std
n_gop = (gop-gop_avg)/gop_std
n_medincome = (medincome-medincome_avg)/medincome_std
n_migrarate = (migrarate-migrarate_avg)/migrarate_std
n_brate = (brate-brate_avg)/brate_std
n_drate = (drate-drate_avg)/drate_std
n_growthrate = (growthrate-growthrate_avg)/growthrate_std
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
t_growthrate = t_brate - t_drate  # New Feature showing population growth rate
t_bachrate = pd.DataFrame(tframe[['BachelorRate']]).to_numpy()  # percent of people with bachelors degree 
t_unemployrate = pd.DataFrame(tframe[['UnemploymentRate']]).to_numpy()  # percent of labor force unemployed

# averages of all features
t_medincome_avg = np.average(t_medincome)
t_migrarate_avg = np.average(t_migrarate)
t_brate_avg = np.average(t_brate)
t_drate_avg = np.average(t_drate)
t_growthrate_avg = np.average(t_growthrate)
t_bachrate_avg = np.average(t_bachrate)
t_unemployrate_avg = np.average(t_unemployrate)

# std of all features
t_medincome_std = np.std(t_medincome)
t_migrarate_std = np.std(t_migrarate)
t_brate_std = np.std(t_brate)
t_drate_std = np.std(t_drate)
t_growthrate_std = np.std(t_growthrate)
t_bachrate_std = np.std(t_bachrate)
t_unemployrate_std = np.std(t_unemployrate)

# Z-score standardized t_features, based on (xi-avg(x))/std(x)
n_t_medincome = (t_medincome-t_medincome_avg)/t_medincome_std
n_t_migrarate = (t_migrarate-t_migrarate_avg)/t_migrarate_std
n_t_brate = (t_brate-t_brate_avg)/t_brate_std
n_t_drate = (t_drate-t_drate_avg)/t_drate_std
n_t_growthrate = (t_growthrate-t_growthrate_avg)/t_growthrate_std
n_t_bachrate = (t_bachrate-t_bachrate_avg)/t_bachrate_std
n_t_unemployrate = (t_unemployrate-t_unemployrate_avg)/t_unemployrate_std

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




# INITIAL PROCESSING 

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

tfeatures2016 = np.dstack((n_t_medincome, n_t_migrarate, n_t_brate, n_t_drate, n_t_growthrate, n_t_bachrate, n_t_unemployrate)).squeeze()# SUBMISSION FEATURE MATRIX
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




