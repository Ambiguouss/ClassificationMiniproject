import numpy as np
from model import *

class SVM(Model):
    def train(self,trainingX,trainingY):
        Y=np.where(trainingY==0,-1,1)