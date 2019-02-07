import json
import numpy as np


###### Q5.1 ######
def objective_function(X, y, w, lamb):
    """
    Inputs:
    - Xtrain: A 2 dimensional numpy array of data (number of samples x number of features)
    - ytrain: A 1 dimensional numpy array of labels (length = number of samples )
    - w: a numpy array of D elements as a D-dimension vector, which is the weight vector and initialized to be all 0s
    - lamb: lambda used in pegasos algorithm

    Return:
    - train_obj: the value of objective function in SVM primal formulation
    """
    
    add=0
    #print ("Shapes::",np.shape(w)[0],X.shape,y.shape)
    w=w.reshape((np.shape(w)[0],))
    #print ("reshaped::",w.shape)
    for i in range(0,len(X)):
      mul1=np.dot(w.transpose(),X[i])
      mul2=np.dot(y[i],mul1)
      diff=1-mul2
      maximum=np.maximum(0,diff)
      add +=maximum
    add=add/len(X)
    norm=np.linalg.norm(w)
    first=(lamb/2)*(norm *norm)
    obj_value=first+add
    
    return obj_value


###### Q5.2 ######
def pegasos_train(Xtrain, ytrain, w, lamb, k, max_iterations):
    """
    Inputs:
    - Xtrain: A list of num_train elements, where each element is a list of D-dimensional features.
    - ytrain: A list of num_train labels
    - w: a numpy array of D elements as a D-dimension vector, which is the weight vector and initialized to be all 0s
    - lamb: lambda used in pegasos algorithm
    - k: mini-batch size
    - max_iterations: the maximum number of iterations to update parameters

    Returns:
    - learnt w
    - traiin_obj: a list of the objective function value at each iteration during the training process, length of 500.
    """
    np.random.seed(0)
    Xtrain = np.array(Xtrain)
    ytrain = np.array(ytrain)
    N = Xtrain.shape[0]
    D = Xtrain.shape[1]

    train_obj = []
    #print ("Shapes TRainn::",w.shape,Xtrain.shape,ytrain.shape)
    for iter in range(1, max_iterations + 1):
        A_t = np.floor(np.random.rand(k) * N).astype(int)  # index of the current mini-batch
        A_t_plus=[]
        
        w_half=np.zeros(w.shape)
        
        for index in A_t:
         
         p=np.dot(w.transpose(),Xtrain[index])
         product=ytrain[index]*p
         #print ("prod::",product)
         if product<1:
            A_t_plus.append(index)
        eta=float(1/(lamb*iter))
        
        sum=np.zeros((np.shape(w)[0],))
        #print ("AT+",len(A_t_plus))
        for index in A_t_plus:
         sum +=np.dot(ytrain[index],Xtrain[index])
        sum=(np.dot(eta,sum))/k
        sum=np.reshape(sum,w.shape)
        first_term=(1-(eta*lamb)) *w
        first_term=np.reshape(first_term,w.shape)
        
        w_half=np.add(first_term,sum)
        
        w_norm=np.linalg.norm(w_half)
        numerator=1/(np.sqrt(lamb))
        minimization=numerator/w_norm
        
        minimum=np.minimum(1,minimization)
        w=np.dot(minimum,w_half)
        obj_val=objective_function(Xtrain,ytrain,w,lamb)
        
        train_obj.append(obj_val)
   
    return w, train_obj


###### Q5.3 ######
def pegasos_test(Xtest, ytest, w, t = 0.):
    """
    Inputs:
    - Xtest: A list of num_test elements, where each element is a list of D-dimensional features.
    - ytest: A list of num_test labels
    - w_l: a numpy array of D elements as a D-dimension vector, which is the weight vector of SVM classifier and learned by pegasos_train()
    - t: threshold, when you get the prediction from SVM classifier, it should be real number from -1 to 1. Make all prediction less than t to -1 and otherwise make to 1 (Binarize)

    Returns:
    - test_acc: testing accuracy.
    """
    # you need to fill in your solution here
    count=0
    Xtest=np.array(Xtest)
    ytest=np.array(ytest)
    w=w.reshape((np.shape(w)[0],))
    pred=np.dot(Xtest,w)
    
    for i in range(0,len(ytest)):
      if pred[i]<t:
         pred[i]=-1
      else:
         pred[i]=1
      if pred[i]==ytest[i]:
         count +=1
  
    test_acc=count/len(Xtest)
    #print ("acc::",test_acc)
    

    return test_acc


"""
NO MODIFICATIONS below this line.
You should only write your code in the above functions.
"""

def data_loader_mnist(dataset):

    with open(dataset, 'r') as f:
            data_set = json.load(f)
    train_set, valid_set, test_set = data_set['train'], data_set['valid'], data_set['test']

    Xtrain = train_set[0]
    ytrain = train_set[1]
    Xvalid = valid_set[0]
    yvalid = valid_set[1]
    Xtest = test_set[0]
    ytest = test_set[1]

    ## below we add 'one' to the feature of each sample, such that we include the bias term into parameter w
    Xtrain = np.hstack((np.ones((len(Xtrain), 1)), np.array(Xtrain))).tolist()
    Xvalid = np.hstack((np.ones((len(Xvalid), 1)), np.array(Xvalid))).tolist()
    Xtest = np.hstack((np.ones((len(Xtest), 1)), np.array(Xtest))).tolist()

    for i, v in enumerate(ytrain):
        if v < 5:
            ytrain[i] = -1.
        else:
            ytrain[i] = 1.
    for i, v in enumerate(ytest):
        if v < 5:
            ytest[i] = -1.
        else:
            ytest[i] = 1.

    return Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest


def pegasos_mnist():

    test_acc = {}
    train_obj = {}

    Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest = data_loader_mnist(dataset = 'mnist_subset.json')

    max_iterations = 500
    k = 100
    for lamb in (0.01, 0.1, 1):
        w = np.zeros((len(Xtrain[0]), 1))
        w_l, train_obj['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_train(Xtrain, ytrain, w, lamb, k, max_iterations)
        test_acc['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_test(Xtest, ytest, w_l)

    lamb = 0.1
    for k in (1, 10, 1000):
        w = np.zeros((len(Xtrain[0]), 1))
        w_l, train_obj['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_train(Xtrain, ytrain, w, lamb, k, max_iterations)
        test_acc['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_test(Xtest, ytest, w_l)

    return test_acc, train_obj


def main():
    test_acc, train_obj = pegasos_mnist() # results on mnist
    print('mnist test acc \n')
    for key, value in test_acc.items():
        print('%s: test acc = %.4f \n' % (key, value))

    with open('pegasos.json', 'w') as f_json:
        json.dump([test_acc, train_obj], f_json)


if __name__ == "__main__":
    main()
