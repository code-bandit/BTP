import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np


df = pd.read_csv('iris.csv')
df = df.drop(['Id'],axis=1)
target = df['Species']
s = set()
for val in target:
    s.add(val)
s = list(s)
rows = list(range(100,150))
df = df.drop(df.index[rows])

## Plot the dataset
# x = df['SepalLengthCm']
# y = df['PetalLengthCm']

# setosa_x = x[:50]
# setosa_y = y[:50]

# versicolor_x = x[50:]
# versicolor_y = y[50:]

# plt.figure(figsize=(8,6))
# plt.scatter(setosa_x,setosa_y,marker='+',color='green')
# plt.scatter(versicolor_x,versicolor_y,marker='_',color='red')
# plt.show()


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
X = df.values.tolist()


## Shuffle and split the data into training and test set
X, Y = shuffle(X,Y)
x_train = []
y_train = []
x_test = []
y_test = []

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.9)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

# y_train = y_train.reshape(90,1)
# y_test = y_test.reshape(10,1)
print(len(x_train[0]))
print(y_train)

#SVM

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

clf = SVC(kernel='rbf', gamma=1.4, C=5.1)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
print(y_pred)
print(accuracy_score(y_test,y_pred))