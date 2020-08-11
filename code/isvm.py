import numpy as np
import pandas as pd
from sklearn import datasets 
from sklearn.model_selection import train_test_split
from pso_isvm import mainPSO

""" Definitions """
T = None # training dataset
K = 3 # no. of classifications
T_m = 1 # mth sample set

""" Decision Function """
def func(x):
    pass

""" Main Function """
def main(k=3):

    iris = datasets.load_iris() # loading the database 

    # X = iris.data # X -> features
    y = iris.target # y -> label 

    y[85] = 2
    y[86] = 2
    y[87] = 2
    y[88] = 2
    y[89] = 2
    y[90] = 2

    df = pd.read_csv('iris.csv')
    df = df.drop(['Id'],axis=1)
    target = df['Species']
    s = set()
    for val in target:
        s.add(val)
    s = list(s)
    rows = list(range(100,150))
    df = df.drop(df.index[rows])

    ## Drop rest of the features and extract the target values
    df = df.drop(['SepalWidthCm','PetalWidthCm'],axis=1)
    Y = []
    target = df['Species']
    for val in target:
        if(val == 'Iris-setosa'):
            Y.append(-1)
        else:
            Y.append(1)
    df = df.drop(['Species'],axis=1)
    x = df.values.tolist()
    X = np.array(x)
    print(y[:len(X)])
    X_train, X_test, y_train, y_test = train_test_split(X, y[:len(X)], train_size=0.9)
    m = 0
    print("x_train : ", (X_train.shape))
    print("y_train : ", (y_train.shape))
    Classifiers = np.array([])
    while m <= k-1:
        num = 0
        for i in range(len(y_train)):
            if y_train[i] == m:
                num = num+1
        
        print("num : ", num)
        newX_train = np.zeros(shape=[num, 2])
        newY_train = np.zeros(shape=[num, ])
        num = 0
        for i in range(len(y_train)):
            if y_train[i] == m:
                newX_train[num, :] = X_train[i]
                newY_train[num] = y_train[i]
                num = num+1
        print("New X : ", (newX_train), len(newX_train))
        print("New Y : ", (newY_train), len(newY_train))
        Classifiers = np.append(Classifiers, mainPSO(newX_train, newY_train))
        m = m+1
    
    print(Classifiers)

if __name__ == "__main__":
    main()