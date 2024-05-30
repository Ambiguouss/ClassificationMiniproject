import numpy as np
import os
from RandomForest import *

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
training_path= os.path.join(project_dir, "dataset", "training.data")
validation_path = os.path.join(project_dir, "dataset", "validation.data")
test_path = os.path.join(project_dir, "dataset", "test.data")

training = np.loadtxt(training_path)
validation = np.loadtxt(validation_path)
test = np.loadtxt(test_path)
trainingX=training[:,:-1]
trainingY=training[:,-1]
validationX=validation[:,:-1]
validationY=validation[:,-1]
testX=test[:,:-1]
testY=test[:,-1]

forest=RandomForest()
forest.train(trainingX,trainingY,2000,2,"entrophy")
print(forest.accuracy(trainingX,trainingY))
print(forest.accuracy(validationX,validationY))
#tree=Tree()
#tree.train(trainingX,trainingY,maxdepth=40)
#print(tree.accuracy(trainingX,trainingY))
#print(tree.accuracy(validationX,validationY))
