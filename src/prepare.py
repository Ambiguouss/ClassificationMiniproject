import numpy as np
import os
from split import *

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_dir, "dataset", "phishing.data")
training_path= os.path.join(project_dir, "dataset", "training.data")
validation_path= os.path.join(project_dir, "dataset", "validation.data")
test_path = os.path.join(project_dir, "dataset", "test.data")
input = np.loadtxt(data_path,delimiter=',')

scale(input)

training,validation,test=split(input)

np.savetxt(training_path,training)
np.savetxt(validation_path,validation)
np.savetxt(test_path,test)

