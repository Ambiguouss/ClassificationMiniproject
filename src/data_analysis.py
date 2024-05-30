import numpy as np
import os
import matplotlib.pyplot as plt

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
training_path= os.path.join(project_dir, "dataset", "training.data")

data = np.loadtxt(training_path)

features = data.shape[1]-1

print(features)

analysis=[]

for i in range(features):
    c2=[]
    for x in range(3):
        c1=[]
        for res in range(2):
            condition = (data[:, i] == x) & (data[:, -1] == res)
            count = np.sum(condition)
            c1.append(count)
        c2.append(c1)
    analysis.append(c2)

print(analysis[0])


