from __future__ import print_function
import json
import numpy as np
import sys

def forward(pi, A, B, O):
  """
  Forward algorithm

  Inputs:
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - alpha: A numpy array alpha[j, t] = P(Z_t = s_j, x_1:x_t)
  """
  S = len(pi)
  N = len(O)
  alpha = np.zeros([S, N])
  
  
  initial=O[0]
  for j in range(len(alpha)):
   alpha[j,0]=pi[j]*B[j,initial]
  #print (" after Initial alpha::",alpha)
  
  for t in range(1,len(O)):
   observation=O[t]
   for j in range(len(alpha)):
      #print ("j::",j)
      prob=0
      for i in range(len(alpha)):
         #print ("i::",alpha[i,t-1])
         prob +=A[i,j]*alpha[i,t-1]
      alpha[j,t]=prob*B[j,observation]
  #print (" Alpha::",alpha)
  ###################################################
  # Q3.1 Edit here
  ###################################################

  return alpha


def backward(pi, A, B, O):
  """
  Backward algorithm

  Inputs:
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - beta: A numpy array beta[j, t] = P(Z_t = s_j, x_t+1:x_T)
  """
  S = len(pi)
  N = len(O)
  beta = np.zeros([S, N])
  ###################################################
  # Q3.1 Edit here
  ###################################################
  for j in range(0,len(beta)):
   beta[j,len(O)-1]=1

  for t in range(N-2,-1,-1):
   observation=O[t+1]
   for i in range(0,S):
      prob=0
      for j in range(0,S):
         prob+=beta[j,t+1]*A[i,j]*B[j,observation]
      beta[i,t]=prob
  #print ("Beta::",beta)
  return beta

def seqprob_forward(alpha):
  """
  Total probability of observing the whole sequence using the forward algorithm

  Inputs:
  - alpha: A numpy array alpha[j, t] = P(Z_t = s_j, x_1:x_t)

  Returns:
  - prob: A float number of P(x_1:x_T)
  """
  prob = 0
  #print ("length::",len(alpha[0]))
  ###################################################
  # Q3.2 Edit here
  ###################################################
  for j in range(0,len(alpha)):
   prob+=alpha[j,len(alpha[0])-1]
  
  return prob


def seqprob_backward(beta, pi, B, O):
  """
  Total probability of observing the whole sequence using the backward algorithm

  Inputs:
  - beta: A numpy array beta: A numpy array beta[j, t] = P(Z_t = s_j, x_t+1:x_T)
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence
      (in terms of the observation index, not the actual symbol)

  Returns:
  - prob: A float number of P(x_1:x_T)
  """
  prob = 0
  
  ###################################################
  # Q3.2 Edit here
  ###################################################
  for i in range(0,len(beta)):
   prob+=beta[i,0]*pi[i]*B[i,O[0]]
  return prob

def viterbi(pi, A, B, O):
  """ 
  Viterbi algorithm

  Inputs:
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - path: A list of the most likely hidden state path k* (in terms of the state index)
    argmax_k P(s_k1:s_kT | x_1:x_T)
  """
  path = []
  S = len(pi)
  N = len(O)
  alpha_1 = np.zeros([S, N])
  
  states=[[0 for j in range(len(O))]for i in range(len(A))]
  
  #print ("states::",states)
  initial=O[0]
  
  for j in range(len(A)):
   alpha_1[j,0]=pi[j]*B[j,initial]
   states[j][0]=j
  #print (" after Initial alpha::",alpha)
  #print ("states::",states)
  for t in range(1,len(O)):
   observation=O[t]
   for j in range(len(A)):
      #print ("j::",j)
      max_prob=0
      for i in range(len(A)):
         #print ("i::",alpha[i,t-1])
         prob=A[i,j]*alpha_1[i,t-1]
         if prob>max_prob:
            max_prob=prob
            max_state=i
      states[j][t]=max_state
      alpha_1[j,t]=max_prob*B[j,observation]
  #print ("states::",states)
  
  maxi=0
  for j in range(0,len(A)):
   if alpha_1[j,N-1]>maxi:
      maxi=alpha_1[j,N-1]
      final_state=j
  for j in range(0,len(A)):
   if j==final_state:
      for i in range(0,len(states[j])):
         path.append(states[j][i])
  ###################################################
  # Q3.3 Edit here
  ###################################################
  
  return path


##### DO NOT MODIFY ANYTHING BELOW THIS ###################
def main():
  model_file = sys.argv[1]
  Osymbols = sys.argv[2]

  #### load data ####
  with open(model_file, 'r') as f:
    data = json.load(f)
  A = np.array(data['A'])
  B = np.array(data['B'])
  pi = np.array(data['pi'])
  #### observation symbols #####
  obs_symbols = data['observations']
  #### state symbols #####
  states_symbols = data['states']

  N = len(Osymbols)
  O = [obs_symbols[j] for j in Osymbols]

  alpha = forward(pi, A, B, O)
  beta = backward(pi, A, B, O)

  prob1 = seqprob_forward(alpha)
  prob2 = seqprob_backward(beta, pi, B, O)
  print('Total log probability of observing the sequence %s is %g, %g.' % (Osymbols, np.log(prob1), np.log(prob2)))

  viterbi_path = viterbi(pi, A, B, O)

  print('Viterbi best path is ')
  for j in viterbi_path:
    print(states_symbols[j], end=' ')

if __name__ == "__main__":
  main()
