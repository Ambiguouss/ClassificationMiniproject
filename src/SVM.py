import numpy as np
from model import *

class SVM(Model):

    def __init__(self,C=1.0,eps=0.001,kernel="linear"):
        self.w=None
        self.b=0
        self.C=C
        self.kernel_name=kernel
        self.eps=eps

    def kernel(self,X,Y):
        method=getattr(self,self.kernel_name,None)
        if callable(method):
            return method(X,Y)
        else: raise AttributeError(f"'{self.__class__.__name__}' no method '{self.method_name}'")

    def linear(self,X,Y):
        return np.dot(X,Y)

    def gauss(self,X,Y):
        distance = np.linalg.norm(X-Y) ** 2
        return np.exp(-distance)
    def sigmoid(self,X,Y):
        return np.tanh(np.dot(X,Y))

    def takeStep(self,i,j,X,Y,alphas):
        if i==j:
            return 0
        a1=alphas[i]
        a2=alphas[j]
        y1=Y[i]
        y2=Y[j]
        E1=self.predict(X[i])-y1
        E2=self.predict(X[j])-y2
        s=y1*y2
        L=np.where(s==-1,max(0,a2-a1),max(0,a2+a1-self.C))
        H=np.where(s==-1,min(self.C,self.C+a2-a1),min(self.C,a2+a1))
        if L==H:
            return 0
        k11=self.kernel(X[i],X[i])
        k22=self.kernel(X[j],X[j])
        k12=self.kernel(X[i],X[j])
        eta=k11+k22-2*k12
        if eta==0:
            return 0
        a2new=a2+y2*(E1-E2)/eta
        if a2new < L:
            a2=L
        elif a2new > H:
            a2=H
        if np.abs(a2new-a2)<self.eps:
            return 0
        a1new=a1+s*(a2-a2new)
        alphas[i]=a1new
        alphas[j]=a2new
        return 1

        


    def updatewb(self,X,Y,alphas):
        self.w=X.T@(alphas*Y)
        for i,a in enumerate(alphas):
            if 0<a<self.C:
                self.b=Y[i]-X[i]@self.w
                break

    def examineExample(self,i,X,Y,alphas):
        y=Y[i]
        a=alphas[i]
        for j in range(len(alphas)):
            if self.takeStep(i,j,X,Y,alphas):
                self.updatewb(X,Y,alphas)
                return 1
        return 0



    def train(self,X,trY,max_iter=10):
        Y=np.where(trY==0,-1,1)
        m=len(Y)
        n=X.shape[0]
        alphas=np.zeros(m)
        self.updatewb(X,Y,alphas)
        numChanged=0
        examineAll=1
        cnt=0
        while (numChanged>0 or examineAll)and (max_iter is None or cnt<max_iter):
            cnt+=1
            numChanged=0
            if examineAll:
                for I in range(m):
                    numChanged+=self.examineExample(I,X,Y,alphas)
            else:
                for I in range(m):
                    if 0<alphas[I]<self.C:
                        numChanged+=self.examineExample(I,X,Y,alphas)
            if examineAll==1:
                examineAll=0
            elif numChanged==0:
                examineAll=1

        



    def predict(self,X):
        return np.dot(X,self.w)+self.b>0