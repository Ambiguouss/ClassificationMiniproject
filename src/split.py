import numpy as np


#only for convience
def scale(input):
    input[:,:21]+=1
    input[:,:21]/=2
    input[:,21:29]+=1
    input[:,-1]+=1
    input[:,-1]/=2


def split(input,train=0.6,valid=0.2):
    input = input[np.argsort(input[:,-1])]
    no_plus = np.sum(input[:,-1]==1)
    no_minus = np.sum(input[:,-1]==0)
    minus = input[:no_minus]
    plus = input[no_minus:]
    np.random.shuffle(minus)
    np.random.shuffle(plus)
    minus_training,minus_valid,minus_test=np.split(minus,[(int)(no_minus*train),(int)(no_minus*(train+valid))])
    plus_training,plus_valid,plus_test=np.split(plus,[(int)(no_plus*train),(int)(no_plus*(train+valid))])
    
    training=np.concatenate((minus_training,plus_training),axis=0)
    validation=np.concatenate((minus_valid,plus_valid),axis=0)
    test=np.concatenate((minus_test,plus_test),axis=0)
    np.random.shuffle(training)
    np.random.shuffle(validation)
    np.random.shuffle(test)
    return (training,validation,test)

