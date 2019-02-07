"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is compute_distances, predict_labels, compute_accuracy
and find_best_k.
"""

import numpy as np
import json


###### Q5.1 ######
def compute_distances(Xtrain, X):
	"""
	Compute the distance between each test point in X and each training point
	in Xtrain.
	Inputs:
	- Xtrain: A numpy array of shape (num_train, D) containing training data
	- X: A numpy array of shape (num_test, D) containing test data.
	Returns:
	- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
	  is the Euclidean distance between the ith test point and the jth training
	  point.
	"""
	#print ("Test data::",Xtrain)
	size_test = len(X)
	size_train= len(Xtrain)
	s=np.shape(X)
	s1=np.shape(Xtrain)
	
	dists=[[0 for x in range(size_train)] for y in range(size_test)] 
	for i in range(0,len(X)):
	   a=np.array(X[i])
	  
	   for j in range(0,len(Xtrain)):
	      b=np.array(Xtrain[j])
	      dist=np.linalg.norm(a-b)
	      dists[i][j]=dist
	   
	dists=np.array(dists)
	#print ("900 900", dists[900,900], dists[700,500] ,dists[10,60])
	
	shape=np.shape(dists)
	#print("shape::",shape)

	 
	return dists

###### Q5.2 ######
def predict_labels(k, ytrain, dists):
	"""
	Given a matrix of distances between test points and training points,
	predict a label for each test point.
	Inputs:
	- k: The number of nearest neighbors used for prediction.
	- ytrain: A numpy array of shape (num_train,) where ytrain[i] is the label
	  of the ith training point.
	- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
	  gives the distance betwen the ith test point and the jth training point.
	Returns:
	- y: A numpy array of shape (num_test,) containing predicted labels for the
	  test data, where y[i] is the predicted label for the test point X[i]. 
	"""
	ypred=[0 for x in range(len(ytrain))]
	
	for i in range(0,len(dists)):
	   sorted_arr=dists[i].argsort()[:k]
	   temp=[]
	   vote=dict()
	   for j in range(0,len(sorted_arr)):
	      label=ytrain[sorted_arr[j]]
	      
	      if(label in vote.keys()):
	         vote[label] +=1
	      elif (label not in vote.keys()):
	         vote[label] =1
	   min_key=100
	   max_vote=0
	   for key in vote.keys():
	      if vote[key]>max_vote:
	         max_vote=vote[key]
	         min_key=key
	      elif vote[key]==max_vote and key<min_key:
	         min_key=key
	    
	   ypred[i]=min_key

	ypred=np.array(ypred)
	shape=np.shape(ypred)
	return ypred

###### Q5.3 ######
def compute_accuracy(y, ypred):
	"""
	Compute the accuracy of prediction based on the true labels.
	Inputs:
	- y: A numpy array with of shape (num_test,) where y[i] is the true label
	  of the ith test point.
	- ypred: A numpy array with of shape (num_test,) where ypred[i] is the 
	  prediction of the ith test point.
	Returns:
	- acc: The accuracy of prediction.
	"""
	count=0
	for i in range(0,len(y)):
	   if(y[i]==ypred[i]):
	      count +=1
	acc=count/float(len(y))
	
	return acc

###### Q5.4 ######
def find_best_k(K, ytrain, dists, yval):
	"""
	Find best k according to validation accuracy.
	Inputs:
	- K: A list of ks.
	- ytrain: A numpy array of shape (num_train,) where ytrain[i] is the label
	  of the ith training point.
	- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
	  is the Euclidean distance between the ith test point and the jth training
	  point.
	- yval: A numpy array with of shape (num_val,) where y[i] is the true label
	  of the ith validation point.
	Returns:
	- best_k: The k with the highest validation accuracy.
	- validation_accuracy: A list of accuracies of different ks in K.
	"""
	
	validation_accuracy=[]
	acc=0.0
	for i in range(0,len(K)):
	   ypred=predict_labels(K[i], ytrain, dists)
	   accuracy=compute_accuracy(yval, ypred)
	   if accuracy>acc:
	      best_k=K[i]
	      acc=accuracy
	   validation_accuracy.append(accuracy)
	   
	return best_k, validation_accuracy


"""
NO MODIFICATIONS below this line.
You should only write your code in the above functions.
"""

def data_processing(data):
	train_set, valid_set, test_set = data['train'], data['valid'], data['test']
	Xtrain = train_set[0]
	ytrain = train_set[1]
	Xval = valid_set[0]
	yval = valid_set[1]
	Xtest = test_set[0]
	ytest = test_set[1]
	
	Xtrain = np.array(Xtrain)
	Xval = np.array(Xval)
	Xtest = np.array(Xtest)
	
	ytrain = np.array(ytrain)
	yval = np.array(yval)
	ytest = np.array(ytest)
	
	return Xtrain, ytrain, Xval, yval, Xtest, ytest
	
def main():
	input_file = 'mnist_subset.json'
	output_file = 'knn_output.txt'

	with open(input_file) as json_data:
		data = json.load(json_data)
	
	#==================Compute distance matrix=======================
	K=[1, 3, 5, 7, 9]	
	
	Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing(data)
	
	dists = compute_distances(Xtrain, Xval)
	
	#===============Compute validation accuracy when k=5=============
	k = 5
	ypred = predict_labels(k, ytrain, dists)
	acc = compute_accuracy(yval, ypred)
	print("The validation accuracy is", acc, "when k =", k)
	
	#==========select the best k by using validation set==============
	best_k,validation_accuracy = find_best_k(K, ytrain, dists, yval)

	
	#===============test the performance with your best k=============
	dists = compute_distances(Xtrain, Xtest)
	ypred = predict_labels(best_k, ytrain, dists)
	test_accuracy = compute_accuracy(ytest, ypred)
	
	#====================write your results to file===================
	f=open(output_file, 'w')
	for i in range(len(K)):
		f.write('%d %.3f' % (K[i], validation_accuracy[i])+'\n')
	f.write('%s %.3f' % ('test', test_accuracy))
	f.close()
	
if __name__ == "__main__":
	main()
