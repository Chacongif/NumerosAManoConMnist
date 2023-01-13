#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import time
import pickle
from csv import writer
from csv import reader
from sklearn.svm import SVC



data = np.loadtxt('mnist_train_copiacolor.csv', delimiter=',') 
df = pd.read_csv('mnist_train_copiacolor.csv')
print("Lectura de la base de datos completa")
ncol = data.shape[1]
X = data[:,1:ncol]
y = data[:,0]


# In[14]:


i = 0
for i in range(ncol):
    if y[i] <= 3:
        color = 1
        df = df.assign[color]
    else:
        color = 0
        df = df.assign[color]

tic = time.process_time()
clf = LogisticRegression()
clf.fit(X, y)
toc = time.process_time()
print("Entrenamiento completo")
print("Tiempo de procesador para el entrenamiento (seg):")
print(toc - tic)
        

data = np.loadtxt('mnist_test_copiacolor.csv', delimiter=',') 
#print(data)
ncol = data.shape[1]
print(ncol)
# definiendo entradas y salidas
X_test = data[:,1:ncol]
y_test = data[:,0]

# predecir los valores de XF_test
predicted = clf.predict(X_test)

# para finalizar se calcula el error
error = 1 - accuracy_score(y_test, predicted)
print(error)

# el modelo entrenado se salva en disco
filename = 'finalized_modelPruebaColor.sav'
pickle.dump(clf, open(filename, 'wb'))


# In[ ]:


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt



        
data = np.loadtxt('mnist_test_copiaColor.csv', delimiter=',') 
ncol = data.shape[1]
X_test = data[:,1:ncol]
y_test = data[:,0]
print(X_test)
print(y_test)


select_image = np.random.random([28, 28])
ex = np.random.randint(0, 10000)
loaded_model1 = pickle.load(open('finalized_modelPruebaColor.sav', 'rb'))


xtest = X_test[ex,].reshape(1, -1)
predicted = loaded_model1.predict(xtest)
print("La regresion logistica predice un:")	
print(predicted)


xtest = X_test[ex,].reshape(1, -1)
print("La red neuronal predice un:")	
print(predicted)


select_image1 = 1 - X_test[ex,] / 255
k = 0
for i in range(0,28):
    for j in range(0,28):
        select_image[i,j] = select_image1[k]
        print(X_test[k])
        k = k + 1     
        
    if predicted > 3:
        plt.imshow(select_image, cmap='Oranges', interpolation='nearest')
        plt.show()
    else:
        plt.imshow(select_image, cmap='Blues', interpolation='nearest')
        plt.show()

