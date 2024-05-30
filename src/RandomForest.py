import numpy as np
from model import *

class Tree(Model):
    class Node:
        def __inti__(self):
            self.feature=None
            self.val=None
            self.left=None
            self.right=None
            self.ret=0

    def __init__(self,inpurity_function="class_error"):
        self.root=self.Node()
        self.inpurity_function=inpurity_function

    def inpurity(self,Y):
        method=getattr(self,self.inpurity_function,None)
        if callable(method):
            return method(Y)
        else: raise AttributeError(f"'{self.__class__.__name__}' no method '{self.method_name}'")

    def class_error(self,Y):
        p = np.sum(Y)/len(Y)
        return 1-np.maximum(p,1-p)
    def gini(self,Y):
        p = np.sum(Y)/len(Y)
        return p*(1-p)
    def entrophy(self,Y):
        p=np.sum(Y)/len(Y)
        if p==0 or p==1:
            return 0
        return -p*np.log(p)-(1-p)*np.log(1-p)

    def best_split(self,X,Y):
        best_inpurity=np.inf
        best_feature=None
        best_val=None
        for feature in range(X.shape[1]):
            for val in range(3):
                left=X[:,feature]==val
                right=X[:,feature]!=val
                N0=left.sum()
                N1=right.sum()
                if N0==0 or N1==0:
                    continue
                weighted_inpurity=N0*self.inpurity(Y[left])+N1*self.inpurity(Y[right])
                if weighted_inpurity<best_inpurity:
                    best_inpurity=weighted_inpurity
                    best_feature=feature
                    best_val=val
        return best_feature,best_val

    def build_tree(self,X,Y,maxdepth=10,col_indices=None):
        if maxdepth==0:
            return None
        node=self.Node()

        p=np.sum(Y)/len(Y)
        node.ret=p>0.5
        if p==0 or p==1:
            node.left=None
            node.right=None
            return node
        best_feature,best_val=self.best_split(X,Y)
        if best_feature is None:
            node.left=None
            node.right=None
            return node
        if col_indices is None:
            node.feature=best_feature
        else:    
            node.feature=col_indices[best_feature]
        node.val=best_val
        leftX=X[X[:,best_feature]==best_val]
        leftY=Y[X[:,best_feature]==best_val]
        rightX=X[X[:,best_feature]!=best_val]
        rightY=Y[X[:,best_feature]!=best_val]
        node.left=self.build_tree(leftX,leftY,maxdepth-1,col_indices)
        node.right=self.build_tree(rightX,rightY,maxdepth-1,col_indices)
        return node


    def train(self,X,Y,maxdepth=10,col_indices=None):

        self.root=self.build_tree(X,Y,maxdepth,col_indices)

    def walk_tree(self,test,node):
        if node.left is None or node.right is None:
            return node.ret
        if(test[node.feature]==node.val):
            return self.walk_tree(test,node.left)
        return self.walk_tree(test,node.right)

    def predict(self,test):
        return self.walk_tree(test,self.root)

class RandomForest(Model):
    
    def __init__(self):
        self.trees=[]
        self.samples=[]

    def train(self,trainingX,trainingY,no_trees=100,max_depth=2,inpurity="class_error"):
        n=trainingX.shape[0]
        m=trainingX.shape[1]
        for B in range(no_trees):
            row_indices=np.random.choice(n,size=n,replace=True)
            col_indices=np.random.choice(m,size=(int)(np.sqrt(m)),replace=False)
            sampleX=trainingX[row_indices][:,col_indices]
            sampleY=trainingY[row_indices]
            tree=Tree(inpurity)
            tree.train(sampleX,sampleY,max_depth,col_indices)
            self.trees.append(tree)
            self.samples.append(row_indices)
            
            

            

            
    def predict(self,test):
        cnt=0
        for tree in self.trees:
            cnt+=tree.predict(test)
        cnt= cnt/len(self.trees)
        return cnt>0.5