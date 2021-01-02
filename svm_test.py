
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve, auc, f1_score, precision_score, recall_score
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn import metrics
# import os
# print(os.listdir("../hw2"))

def datahandling(filename):
    data_full=pd.read_csv(filename)
    data_full=data_full.dropna()    #滤除缺失数据的行
    data_full['Class'].replace(0,-1,inplace = True)
    data_fraud=data_full[data_full.Class==1]
    #print(len(data_fraud.index))    #只有492个欺诈数据，分类分布很不均衡
    #print(data_fraud.head(3))
    data_safe=data_full[data_full.Class==-1]
    data_use = pd.concat([data_safe.head(400),data_fraud.head(400)],axis=0,ignore_index=True)
    x_base=data_use.iloc[:,1:29]
    y=data_use.loc[:,['Class']]

    return x_base.to_numpy(), y.to_numpy()

class SVM(object):
    def __init__(self, X, y, C, **kerargs):
        self.X = X
        self.y = y
        self.C = C
        self.m = np.shape(X)[0]
        self.n = np.shape(X)[1]
        self.alpha = np.zeros(self.m,dtype='float64')
        self.b = 0.0
        self.Kmat =  np.zeros((self.m,self.m),dtype='float64')
        self.kargs = kerargs
        self.Elist = np.zeros((self.m,2))
        self.sv = ()
        self.sv_idx = None
        #compute K matrix
        for i in range(self.m):
            for j in range(self.m):
                if self.kargs['kernel'] == 'rbf':
                    sig = self.kargs['sigma']
                    self.Kmat[i,j] = np.exp(sum((self.X[i,:]-self.X[j,:])*(self.X[i,:]-self.X[j,:]))/(-1*sig**2)) 
                elif self.kargs['kernel'] == 'linear':
                    self.Kmat[i,j] = sum(self.X[i,:]*self.X[j,:])

    def j_choose(self, i, Ei):
        temp = 0.0
        j_choose = 0
        Ej_choose = 0.0
        E_list_choose = np.nonzero(self.Elist[:,0])[0]
        if len(E_list_choose)>1:
            for j in E_list_choose:
                if j == i:
                    continue                
                Ej = np.dot(self.alpha*self.y, self.Kmat[:,j]) + self.b - float(self.y[j])
                if abs(Ei-Ej)>temp:
                    j_choose = j
                    temp = abs(Ei-Ej)
                    Ej_choose = Ej
            return j_choose, Ej_choose
        else:
            while (j_choose==i):
                j_choose = random.randint(0,self.m)
            Ej_choose = np.dot(self.alpha*self.y, self.Kmat[:,j_choose]) + self.b - float(self.y[j_choose])
            return j_choose, Ej_choose

    def update_alphapair(self, i):
        Ei = np.dot(self.alpha*self.y, self.Kmat[:,i]) + self.b - float(self.y[i])
        #首先选择不满足KKT条件的点
        if ((self.y[i] * Ei < -0.01) and (self.alpha[i] < self.C)) or ((self.y[i] * Ei > 0.01) and (self.alpha[i] > 0)):
            self.Elist[i] = [1,Ei]
            j, Ej = self.j_choose(i,Ei)
            alphaj_old = self.alpha[j].copy()
            alphai_old = self.alpha[i].copy()
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
            self.Elist[j] = [1,np.dot(self.alpha*self.y, self.Kmat[:,j]) + self.b - float(self.y[j])]
            if abs(alphaj_old-self.alpha[j])<1e-5:
                return 0
            self.alpha[i]=self.alpha[i]+self.y[i]*self.y[j]*(alphaj_old-self.alpha[j])
            self.Elist[i] = [1,np.dot(self.alpha*self.y, self.Kmat[:,i]) + self.b - float(self.y[i])]
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



    def SMO(self):
        iter_num = 0
        flag = True
        alphapair_update_num = 0
        while iter_num < 10000 and (alphapair_update_num >0 or flag):
            alphapair_update_num = 0
            if flag:
                for i in range(self.m):
                    alphapair_update_num = alphapair_update_num+self.update_alphapair(i)
                iter_num += 1
                print('loop1')    
            else:
                nonbound_label = np.nonzero((self.alpha > 0) * (self.alpha < self.C))[0]
                for i in nonbound_label:
                    alphapair_update_num = alphapair_update_num+self.update_alphapair(i)
                iter_num += 1
                print('loop2') 
            if flag:
                flag = False
                print('loop3') 
            elif  alphapair_update_num == 0:
                flag = True
                print('loop4') 
            print('iter ',iter_num)    
        # Model
        self.sv_idx = np.nonzero(self.alpha)[0]
        self.sv = self.X[self.sv_idx]
        self.alpha_sv = self.alpha[self.sv_idx]
        self.y_sv = self.y[self.sv_idx]    
        self.X = None
        self.Kmat = None
        self.y = None
        self.alpha = None
        self.Elist = None

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
            # while y_pre == 0:
            #     y_pre = random.uniform(-1, 1)
            # if y_pre > 0:
            #     y_pre = 1
            # else:
            #     y_pre = -1
            y_test.append(y_pre)                
        return y_test


def main():


    # data = pd.read_table('D:\\Codingpy\\hw2\\testSet.txt', header = None)


    # predictors = data.iloc[:,0:20]
    # predictors = predictors.drop(['pdays'],axis=1)
    # predictors = predictors.drop(['duration'],axis=1)
    # y = data.iloc[:,20]
    # X = pd.get_dummies(predictors)
    # y = y.map({'yes': 1,'no': -1})
    # X = data.iloc[:,0:2]
    # y = data.iloc[:,2]


    X,y=datahandling("D:\\Codingpy\\hw2\\creditcard.csv")
    y = y.flatten()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0,stratify=y)
    # X_s,y_s = SMOTE(random_state=0).fit_resample(X_train,y_train)
    # ssc = StandardScaler()
    # ssc.fit(X_s)
    # X_train_s = ssc.transform(X_s)
    # X_test_s = ssc.transform(X_test)
    # y_train_input = y_s.values
    # y_test_input = y_test.values
    X_train_s = X_train
    X_test_s = X_test
    y_train_input = y_train
    y_test_input = y_test    
    print(X_train_s.shape)
    print(y_train_input.shape)
    print(X_test_s.shape)    
    print(y_test_input.shape)

    model = SVM(X_train_s, y_train_input, 1, kernel='rbf', sigma=20)
    # model = SVM(X_train_s, y_train_input, 1, kernel = 'linear')
    begin = datetime.datetime.now()
    model.SMO()
    end = datetime.datetime.now() 
    train_time = end - begin
    print("keneral_svm time", train_time.total_seconds())
    print(len(model.sv_idx))

    y_testResult = model.predict(X_test_s)
    # print("Guassian kernel- ","Precision: ",round(precision_score(y_test_input,y_testResult),2),"Recall: ",round(recall_score(y_test_input,y_testResult),2))

    plt.figure(1) # 创建图表1mb        
    plt.title('Precision/Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    precision, recall, thresholds = metrics.precision_recall_curve(y_test_input, y_testResult)   #直接用sklearn的PRC和AUC的工具
    plt.figure(1)
    plt.plot(precision, recall)
    
    area = metrics.auc(recall, precision)
    print("AUPRC:",area)
    plt.show()

if __name__=='__main__':
   main()
