import json
import random
import numpy as np


def gmm_clustering(X, K):
    """
    Train GMM with EM for clustering.

    Inputs:
    - X: A list of data points in 2d space, each elements is a list of 2
    - K: A int, the number of total cluster centers

    Returns:
    - mu: A list of all K means in GMM, each elements is a list of 2
    - cov: A list of all K covariance in GMM, each elements is a list of 4
            (note that covariance matrix is symmetric)
    """
    
    

    # Initialization:
    pi = []
    mu = []
    cov = []
    for k in range(K):
        pi.append(1.0 / K)
        mu.append(list(np.random.normal(0, 0.5, 2)))
        temp_cov = np.random.normal(0, 0.5, (2, 2))
        temp_cov = np.matmul(temp_cov, np.transpose(temp_cov))
        cov.append(list(temp_cov.reshape(4)))

    ### you need to fill in your solution starting here ###
################################### E -STEP #########################################################################################
    for t in range(100):
      list_x=[]
      for i in range(len(X)):
         denominator=0
         list_k=[]
         for k in range(K):
            
            cov_matrix=np.array(cov[k]).reshape((2,2))
            
            first_term= np.log(pi[k]) - np.log(2*np.pi* np.sqrt(np.linalg.det(cov_matrix)))
            
            second_term=  1/2 * np.dot((np.subtract(X[i],mu[k]).transpose()), (np.dot(np.linalg.inv(cov_matrix),(np.subtract(X[i],mu[k])))))
            
            numerator=first_term-second_term
            list_k.append(np.exp(numerator))
            denominator +=np.exp(numerator)
         
         list_k=[x/denominator for x in list_k]
         #print ("list is::",list_k)
         list_x.append(list_k)  
         
         
######################## M STEP ################################################################################################
      
      for j in range(len(mu)):
         temp=np.array(list_x).transpose()
   
         add =0
         X=np.array(X)
         for l in range(len(X)):
            add +=temp[j][l] * X[l]
            
         mu[j]=list(add/sum(i[j] for i in list_x))

      mu=np.array(mu)
      for j in range(len(cov)):
         add1=0
         
         for l in range(len(X)):
            difference=X[l]-mu[j]
            difference=difference.reshape((1,2))
            product=np.dot( difference.transpose(),difference)
            
            add1 +=temp[j][l] * product
         
         cov[j]=(add1/sum(i[j] for i in list_x)).reshape((4,)).tolist()
      
      mu=mu.tolist() 
    
      for j in range(0,len(pi)):
      
         pi[j]=sum(i[j] for i in list_x)/np.sum(list_x)
   
    return mu, cov


def main():
    # load data
    with open('hw4_blob.json', 'r') as f:
        data_blob = json.load(f)

    mu_all = {}
    cov_all = {}

    print('GMM clustering')
    for i in range(5):
        np.random.seed(i)
        mu, cov = gmm_clustering(data_blob, K=3)
        mu_all[i] = mu
        cov_all[i] = cov

        print('\nrun' + str(i) + ':')
        print('mean')
        print(np.array_str(np.array(mu), precision=4))
        print('\ncov')
        print(np.array_str(np.array(cov), precision=4))

    with open('gmm.json', 'w') as f_json:
        json.dump({'mu': mu_all, 'cov': cov_all}, f_json)


if __name__ == "__main__":
    main()
