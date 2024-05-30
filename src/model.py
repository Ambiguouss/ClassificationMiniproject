import numpy as np

class Model:
    def train(self,trainingX,trainingY):
        pass
    def predict(self,test):
        pass
    def accuracy(self,testX,testY,th=0.5):
        results = np.apply_along_axis(self.predict,axis=1,arr=testX)
        success = np.count_nonzero(results==testY)
        return success/results.shape[0]
