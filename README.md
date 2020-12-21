# ElectionPredictionClassifier  2020
Data set and model solution for Intro to Machine Learning (CS 4780) final project at Cornell University 

The task in this project is to forecast election results. Economic and sociological factors have been widely used when making predictions on the voting results of US elections. Economic and sociological factors vary a lot among counties in the United States. In addition, as you may observe from the election map of recent elections, neighbor counties show similar patterns in terms of the voting results. In this project you will bring the power of machine learning to make predictions for the county-level election results using Economic and sociological factors and the geographic structure of US counties.

My solution uses a basic and creative solution to develop this classifier. The creative solution utilizes additional locational graph data.

Basic Data: 10 features from train_2016.csv 

FIPS - Federal Information Processing Series, are numeric codes assigned by the National Institute of Standards and Technology (NIST). US counties are identified by a 3-digit number.

County - the name of the county, given by county_name, state_initials.

DEM - number of democratic votes recieved.

GOP - number of republican votes recieved.

MedianIncome - median household income.

MigraRate - the migration rate as the difference between the number of persons entering and leaving a country during the year per 1,000 persons (based on midyear population).

BirthRate - frequency of live births in a given population as live births per 1,000 inhabitants.

DeathRate - the ratio of deaths to the population of a particular area or during a particular period of time, calculated as the number of deaths per one thousand people per year.

BachelorRate - percent of inhabitants with a bachelors degree (who are above some age threshold).

UnemploymentRate - percent of the labor force that is unemployed (who are above some age threshold).

Additional Data: Geographic neighbor structure of counties from graph.csv where each row represents the FIPS Code of two directly connecting neighbor counties.

SRC - The source county node id

DST - The destination county node id


Labels: 

0: The county votes GOP
1: The county votes DEM


To create the classifier, I manipulated and preprocessed including developing additional features. Created a validation, test and training set, and a subsample set of an even amount of positive and negative class data. To create a new feature, I used a k-Nearest-Neighbor algorithm to predict swing states by using the locational graph data on labeled data.   

 I trained both a Deep Neural Net and a soft-margin Support Vector Machine. I engaged in model assessment using k-fold Cross Validation and Grid-Search on learning rate, activation function, number of training epochs, kernel and C-value for error-allowing. Lastly, I optimized the training algorithms, evaluated on the test set, and predicted on the unlabeled test set. 



Basic Explanation:

How did you preprocess the dataset and features?

We collected both train_2016 and test_2016 data using pandas dataframes, turned them into numpy arrays. We edited some of the features to work with them easier like making them ints instead of strings and removed commas. We then created county_winner, a 1x1555 array, from the train_2016 data set by counting all the times dem>gop and made that a 1, with everything else 0. This would function as our labels for the labeled training data. We also created 2 new features (although they were not used in the basic solution) being growthrate and spread. Growthrate as birthrate-deatrate, and the idea was to give the model an easier way to numerically show the average age of a county, as we thought average age of a county (whether young or old) could give the model insight in if it is more right or left swinging. Spread just showed by how much each county won dem or gop. We then normalized the features by creating variables for the max, min, average, and standard deviation for each feature, and used Z-Score normalization (x-avg(x))/std(x). After this, we had all the features from both the train_2016 and test_2016 datasets fully normalized and ready to use as numpy arrays. We then processes these features by combining the normalized feature vectors into a feature matrix train_dataset, and setting county_winner to the labels variable. These would by our features and labels. Of the 1555 examples, 1100 of them would be put into k_xtrain_dataset and k_ytrain_dataset to be used for the k-fold cross validation, and the rest would be used for a test set. We also made tfeatures which is a 1555x6 features matrix for the test_2016 data. 
    

Which two learning methods from class did you choose and why did you made the choices?

We decided to use a soft margin Support Vector Machine and a fully connected Neural Network as our two learning methods. From previous experience and some research, we knew that these to learning algorithms are some of the most commonly used algorithms in ML industry for training on these types of data sets. Assuming the data was not linearly separable, and considering that we wanted to use eager learning as opposed to a lazy learning algorithms, we chose not to use perceptron or kNN. We also wanted to take advantage of an algorithm with easily tunable hyperparamters, so soft margin SVM and NNs stuck out. 
    
How did you do the model selection?

We decided that using cross validation would help to minimize large amounts of over or under fitting, and that grid-search would help to find the optimal hyper parameters. Considering the 6 base features, and 1555 training examples, we assumed a neural network with 2 hidden layers with 10 nodes each and a non-linear activation functions made sense. We did not want the number of nodes to be too large as to jump from a size 6 input to something much larger. Also, to prevent overfitting, we did not want to make the network excessively deep. Therefore, we created a k=10 k-fold cross validation to use on our training set for both the SVM and Neural Net. This took m/k splits of the training set, trained the rest of the training data and tested the accuracy on the validation set, and averaged all the accuracies together. 
    
Additionally, we used grid-search with a list of hyper parameters for both the SVM and NN. For each permutation of hyperparameters, we ran a cross-val and printed the average accuracy. The 3 models that gave the highest accuracy are used to test on the test set to evaluate accuracy. 
    
    

Creative Explanation:

Please explain in detail how you achieved this and what you did specifically and why you tried this.

For the creative solution, we decided to utilize all of our resources and use the 2012 training data as well as the graph locational data for counties. We did two unique things in an attempt to allow our model to predict county voting as accurately as possible.

The first was creating two new features in an attempt to give the model a better understanding of the data. The first was growthrate, simply the birth rate - death rate. Using insight that the average age of a county could very well be a good predictor of voting, since it is fair to assume that younger populations are more dem swinging and older populations are more gop swinging. Therefore, the growthrate feature would help to give the model a more direct way in differentiating between counties with younger families versus older families having less kids. 

Second, we created geoSwing, a feature that would give the model a pre-trained prediction of how likely a county is to vote gop or dem. We all know that the country is ver regional regarding how people vote, with most dem or gop counties voting in clumps (for example, the south is often all red while the northeast is often blue, and counties near cities are usually blue and counties far from cities usually are red). 

Using the 2012_train data, we ran an algorithm very similar to k-Nearest-Neighbor. This involved taking every county from the 2012 data as training points, and running kNN on the 2016 data. Using the graph data, we would take a 2016 county, find all the counties which are connected to that county through the graph data, see if they are labeled in the 2012_train data, and then add a 1 or -1 if it is labeled dem or gop, respectively. If it so happens that the fip county is not in the 2012_train data set, so we have no label, then we just keep 0. We add up all these numbers for the counties that rea close to the 2016 county we are looking at, and the final number is the feature for that example. Infering that the data given in 2012_train is evenly spread throughout the country, this feature gives the model a hint at how likely the example is to be gop or dem. This feature is 1555x1 matrix, where each entry is a number between -4 and 1, 1 being it is close to all dem counties and -4 being close to all gop counties.

In order to get the geoSwing training data for the 2016_test features, we ran a second neural network with the inputs being the 2012 training features and the outputs being geoswing. We then predicted this on the 2012 test features to get the predicted labels for the geoSwing features which would then be a feature in the 2016 test data. 

Overall, the object as to use geographical data to give the model a way to know not just if a county might be more gop or dem, but by what degree. 

We also used a subsampling method by creating a smaller data set made up of an equal number of positively and negatively labeled examples. We used this as part of our cross validation in doing both neural nets. 





 
