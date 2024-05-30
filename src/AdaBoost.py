import numpy as np
from model import *
from RandomForest import Tree

class AdaBoost(Model):
    
    

    def __init__(self,feature_values=3):
        self.weak_learners=[]
        self.feature_values=feature_values
        self.classificators=[]

    def weighted_error(self,M,X,Y,weightsX):
        results = np.apply_along_axis(M.predict,axis=1,arr=X)
        return np.sum(weightsX*(results!=Y))
    
    def train(self,trainingX,trainingY,max_iter=100):
        m=trainingX.shape[0]
        n=trainingX.shape[1]
        weightsX=np.full(m,1/m)
        #create weak learners
        for i in range(n):
            for j in range(self.feature_values):
                t=Tree()
                l=t.Node()
                r=t.Node()
                l.ret=0
                r.ret=1
                l.left=None
                l.right=None
                r.left=None
                r.right=None
                t.root.left=l
                t.root.right=r
                t.root.feature=i
                t.root.val=j
                self.weak_learners.append(t)
        for t in range(max_iter):
            min_error=np.inf
            min_class=None
            for i,M in enumerate(self.weak_learners):
                temp=self.weighted_error(M,trainingX,
                                         trainingY,weightsX)
                if temp<min_error:
                    min_error=temp
                    min_class=M
            #print(min_class.root.feature)
            alpha=1/2*np.log((1-min_error)/min_error)
            #print(alpha)
            Z=2*np.sqrt(min_error*(1-min_error))
            pred=(trainingY==np.apply_along_axis(min_class.predict,
                                axis=1,arr=trainingX))
            pred=pred.astype(int)
            pred*=2
            pred-=1
            weightsX=weightsX/Z*np.exp(-alpha*pred)
            self.classificators.append((min_class,alpha))
        




    def predict(self,test):
        results = 0
        for M,w in self.classificators:
            results+=(M.predict(test)*2-1)*w
        return results>0