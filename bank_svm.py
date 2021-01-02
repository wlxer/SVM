#homework 2
#import everything
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve, auc, f1_score, precision_score, recall_score
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn import metrics

path = os.path.abspath(os.path.dirname(sys.argv[0]))

class SVM(object):
    def __init__(self, X, y, C, Tol, **kerargs):
        #Train set
        self.X = X
        self.y = y
        #Penalty coefficient
        self.C = C
        #KKT Tol
        self.tol = Tol
        #row number
        self.m = np.shape(X)[0]
        #column number
        self.n = np.shape(X)[1]
        #dual parameter
        self.alpha = np.zeros(self.m,dtype='float64')
        #b
        self.b = 0.0
        #kernel matrix
        self.Kmat =  np.zeros((self.m,self.m),dtype='float64')
        #use kargs to choose kernel func
        self.kargs = kerargs
        #Record the label that violates the KKT condition. 1 is the tag
        self.Elist = np.zeros((self.m,2))
        #supoort vector and its label/index
        self.sv = ()
        self.sv_idx = None
        #compute K matrix
        for i in range(self.m):
            for j in range(i+1):
                if self.kargs['kernel'] == 'rbf':
                    sig = self.kargs['sigma']
                    self.Kmat[i,j] = np.exp(sum((self.X[i,:]-self.X[j,:])*(self.X[i,:]-self.X[j,:]))/(-1*sig**2))
                    self.Kmat[j,i] = self.Kmat[i,j] #Avoid double counting
                elif self.kargs['kernel'] == 'linear':
                    self.Kmat[i,j] = sum(self.X[i,:]*self.X[j,:])
                    self.Kmat[j,i] = self.Kmat[i,j] #Avoid double counting
            # print("K mat number",i*self.m)

#When i is fixed, j is choosed according to heuristic method: the absolute value of EI-EJ is maximum
    def j_choose(self, i, Ei):
        temp = 0.0
        j_choose = 0
        Ej_choose = 0.0
        #Point violating KKT condition
        E_list_choose = np.nonzero(self.Elist[:,0])[0]
        if len(E_list_choose)>1:
            for j in E_list_choose:
                if j == i:
                    continue                
                Ej = np.dot(self.alpha*self.y, self.Kmat[:,j]) + self.b - float(self.y[j])
                #Bubble selection method
                if abs(Ei-Ej)>temp:
                    j_choose = j
                    temp = abs(Ei-Ej)
                    Ej_choose = Ej
            return j_choose, Ej_choose
        else:
            #random selection at first
            while (j_choose==i):
                j_choose = random.randint(0,self.m)
            Ej_choose = np.dot(self.alpha*self.y, self.Kmat[:,j_choose]) + self.b - float(self.y[j_choose])
            return j_choose, Ej_choose

#SMO inner iteration
    def update_alphapair(self, i):
        Ei = np.dot(self.alpha*self.y, self.Kmat[:,i]) + self.b - float(self.y[i])
        #the point which does not satisfy the KKT condition is selected
        if ((self.y[i] * Ei < -self.tol) and (self.alpha[i] < self.C)) or ((self.y[i] * Ei > self.tol) and (self.alpha[i] > 0)):
            #Update the list of tags that violate KKT conditions
            self.Elist[i] = [1,Ei]
            j, Ej = self.j_choose(i,Ei)
            alphaj_old = self.alpha[j].copy()
            alphai_old = self.alpha[i].copy()

            #Determine the values of lower bound L and upper bound H of alphaj   
            if self.y[i] != self.y[j]:
                L = max(0, self.alpha[j]-self.alpha[i])
                H = min(self.C, self.C+self.alpha[j]-self.alpha[i])
            else:
                L = max(0, self.alpha[j]+self.alpha[i]-self.C)
                H = min(self.C, self.alpha[j]+self.alpha[i])
            if L == H:
                return 0                
            eta = 2*self.Kmat[i,j]-self.Kmat[i,i]-self.Kmat[j,j]
            if eta >= 0:
                return 0
            self.alpha[j]=self.alpha[j]-self.y[j]*(Ei-Ej)/eta
            if self.alpha[j]>=H:
                self.alpha[j]=H
            elif self.alpha[j]>L:
                self.alpha[j]=self.alpha[j]
            else:
                self.alpha[j]=L

            #Update the list of tags that violate KKT conditions    
            self.Elist[j] = [1,np.dot(self.alpha*self.y, self.Kmat[:,j]) + self.b - float(self.y[j])]

            #If the change is not big, exit the inner loop
            if abs(alphaj_old-self.alpha[j])<1e-5:
                return 0

            #update alpha i
            self.alpha[i]=self.alpha[i]+self.y[i]*self.y[j]*(alphaj_old-self.alpha[j])

            #Update the list of tags that violate KKT conditions 
            self.Elist[i] = [1,np.dot(self.alpha*self.y, self.Kmat[:,i]) + self.b - float(self.y[i])]

            # #update b
            b1 = self.b-Ei-self.y[i]*self.Kmat[i,i]*(self.alpha[i]-alphai_old)-self.y[j]*self.Kmat[i,j]*(self.alpha[j]-alphaj_old)
            b2 = self.b-Ej-self.y[i]*self.Kmat[i,j]*(self.alpha[i]-alphai_old)-self.y[j]*self.Kmat[j,j]*(self.alpha[j]-alphaj_old)
            if  0<self.alpha[i]<self.C:
                self.b=b1
            elif 0<self.alpha[j]<self.C:
                self.b=b2
            else:
                self.b=(b1+b2)/2.0
            return 1                                   
        else: return 0


#SMO outer iteration
    def SMO(self):
        iter_num = 0
        flag = True
        alphapair_update_num = 0
        while iter_num < 10000 and (alphapair_update_num >0 or flag):
            alphapair_update_num = 0
            #Check all alpha first
            if flag:
                for i in range(self.m):
                    alphapair_update_num = alphapair_update_num+self.update_alphapair(i)
                iter_num += 1
                print('loop1') 
            #After all the points are traversed, if there are still alpha pairs that change, 
            # the points that do not meet the KKT condition are first traversed      
            else:
                nonbound_label = np.nonzero((self.alpha > 0) * (self.alpha < self.C))[0]
                for i in nonbound_label:
                    alphapair_update_num = alphapair_update_num+self.update_alphapair(i)
                iter_num += 1
                print('loop2') 
            if flag:
                flag = False
            elif  alphapair_update_num == 0:
                flag = True
            print('iter ',iter_num)    
        
        # Save model parameters
        self.sv_idx = np.nonzero(self.alpha)[0]
        self.sv = self.X[self.sv_idx]
        self.alpha_sv = self.alpha[self.sv_idx]
        self.y_sv = self.y[self.sv_idx]    
        self.X = None
        self.Kmat = None
        self.y = None
        self.alpha = None
        self.Elist = None

#Make predictions on the test set
    def predict(self,X_test):
        X_test = np.array(X_test)
        y_test = []
        for i in range(np.shape(X_test)[0]):
            y_pre = self.b
            for j in range(len(self.sv_idx)):
                Ker = 0.0
                if self.kargs['kernel'] == 'rbf':
                    sig = self.kargs['sigma']
                    Ker = np.exp(sum((X_test[i,:]-self.sv[j])*(X_test[i,:]-self.sv[j]))/(-1*sig**2)) 
                elif self.kargs['kernel'] == 'linear':
                    Ker = sum(X_test[i,:]*self.sv[j])
                y_pre = y_pre + self.alpha_sv[j]*self.y_sv[j]*Ker
            y_test.append(y_pre)                
        return y_test


def main():
#Read the original file and preprocess it
    data = pd.read_csv(path+'/bank-additional-full.csv', sep = ',')
    #it would be better to remove the column 'Pdays', as it is not an appropriate factor to use for predictive model
    data = data.drop(['pdays'],axis=1)
    #follow the description, column 'Duration' should be removed too
    data = data.drop(['duration'],axis=1)
    #Remove rows with missing values
    data = data.dropna()
    #Positive and negative samples were extracted
    data_yes = data[data.y=='yes']
    data_no = data[data.y=='no']
    #Select part of the data for classification
    data_sel = pd.concat([data_yes.head(1000),data_no.head(n=1000)],axis=0,ignore_index=True) #for the case 1,2
    # data_sel = pd.concat([data_yes.head(400),data_no.head(n=1000)],axis=0,ignore_index=True) #for the case 3,4,5,6,7,8
    y = data_sel.iloc[:,18]
    predictors = data_sel.iloc[:,0:18]
    #Converting non numerical data into numerical data
    X = pd.get_dummies(predictors)
    #map the label to 1 and -1
    y = y.map({'yes': 1,'no': -1})
    #check y counts
    print(pd.Series(y).value_counts())

#sample mathod
    # #undersampler
    X_u,y_u = RandomUnderSampler(random_state=0).fit_resample(X,y)
    X_train, X_test, y_train, y_test = train_test_split(X_u, y_u, test_size=0.4, random_state=0, stratify=y_u) #for the case 3,4

    # #oversampler
    # X_o,y_o = RandomOverSampler(random_state=0).fit_resample(X,y)
    # X_train, X_test, y_train, y_test = train_test_split(X_o, y_o, test_size=0.4, random_state=0, stratify=y_o) #for the case 5,6

    # #smotesampler
    # X_s,y_s = SMOTE(random_state=0).fit_resample(X,y)
    # X_train, X_test, y_train, y_test = train_test_split(X_s, y_s, test_size=0.4, random_state=0, stratify=y_s) #for the case 7,8

#data normalization
    ssc = StandardScaler()
    ssc.fit(X_train)
    X_train_s = ssc.transform(X_train)
    X_test_s = ssc.transform(X_test)
    y_train_input = y_train.values
    y_test_input = y_test.values
#check data shape
    print('x train',X_train_s.shape)
    print('y train',y_train.shape)
    print('x test',X_train_s.shape)
    print('y test',y_train.shape)      

#SVM with Linear Kernel Fuction
    model = SVM(X_train_s, y_train_input, 1, 0.0001, kernel = 'linear')
    #Calculate SMO Time
    begin = datetime.datetime.now()
    model.SMO()
    end = datetime.datetime.now() 
    train_time = end - begin
    print("Linear_svm time", train_time.total_seconds())
    print('Linear sv_num',len(model.sv_idx))
    y_testResult = model.predict(X_test_s)
    #use precision_recall_curve fuction
    precision, recall, thresholds = metrics.precision_recall_curve(y_test_input, y_testResult)
    area = metrics.auc(recall, precision)
    print("AUPRC_Linear:",area)

#SVM with RBF Kernel Fuction
    model_rbf = SVM(X_train_s, y_train_input, 1, 0.000001, kernel='rbf', sigma=20)
    #Calculate SMO Time
    begin = datetime.datetime.now()
    model_rbf.SMO()
    end = datetime.datetime.now() 
    train_time = end - begin
    print("RBF_svm time", train_time.total_seconds())
    print('RBF sv_num',len(model_rbf.sv_idx))
    y_testResult_RBF = model_rbf.predict(X_test_s)
    #use precision_recall_curve fuction
    precision_RBF, recall_RBF, thresholds_RBF = metrics.precision_recall_curve(y_test_input, y_testResult_RBF)
    area_rbf = metrics.auc(recall_RBF, precision_RBF)
    print("AUPRC_RBF:",area_rbf)

#Plot Precision/Recall Curve and output AUROC
    plt.figure(1)       
    plt.title('Precision/Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.figure(1)
    plt.plot(precision, recall, color='skyblue', label='Linear kernel curve')
    plt.plot(precision_RBF, recall_RBF, color='black', label='RBF kernel curve')
    plt.legend()
    plt.show()

if __name__=='__main__':
   main()
