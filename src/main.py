import numpy as np
import os
import time
from RandomForest import *
from AdaBoost import *

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

#ada=AdaBoost()
#ada.train(trainingX,trainingY,10)
#print(ada.accuracy(trainingX,trainingY))
#print(ada.accuracy(validationX,validationY))
#forest=RandomForest()
#forest.train(trainingX,trainingY,200,2,"entrophy")
#print(forest.accuracy(trainingX,trainingY))
#print(forest.accuracy(validationX,validationY))
#depth_table=[8]
#no_table=[200]
#bagging_table=[12]
#inpurity_table=["entrophy"]
#for depth in depth_table:
#    for no in no_table:
#        for bag in bagging_table:
#            for inpurity in inpurity_table:
#                forest=RandomForest()
#                start_train_time = time.time()
#                forest.train(trainingX,trainingY,no_trees=no,max_depth=depth,inpurity=inpurity,bagging=bag)
#                end_train_time = time.time()
#                train_time = end_train_time - start_train_time
#                start_predict_time=time.time()
#                acc=forest.accuracy(testX,testY)
#                end_predict_time=time.time()
#                predict_time=end_predict_time-start_predict_time
#                print(f"Tree depth {depth} No trees {no} Bag {bag} Impurity {inpurity}\n Training time: {train_time} seconds\nPredict time: {predict_time}\nAccuracy: {acc}")
iter_table=[50]
for iter in iter_table:
    ada=AdaBoost()
    start_train_time = time.time()
    ada.train(trainingX,trainingY,iter)
    end_train_time = time.time()
    train_time = end_train_time - start_train_time
    start_predict_time=time.time()
    acc=ada.accuracy(testX,testY)
    end_predict_time=time.time()
    predict_time=end_predict_time-start_predict_time
    print(f"Iter {iter}\n Training time: {train_time} seconds\nPredict time: {predict_time}\nAccuracy: {acc}")
