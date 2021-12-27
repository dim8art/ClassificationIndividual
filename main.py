from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import random
import math
import numpy as np
import pandas as pd
import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
TRAIN_SIZE = 700


DATA_X = np.array(pd.read_csv('data2X.csv'))
DATA_Y = np.array(pd.read_csv('data2Y.csv'))
DATA_SIZE = len(DATA_X)

train_x = np.zeros((TRAIN_SIZE, 2))
train_y = np.zeros(TRAIN_SIZE)
for i in range(TRAIN_SIZE):
    n = random.randint(0, DATA_SIZE-1)
    train_x[i][0] = DATA_X[n][0]
    train_x[i][1] = DATA_X[n][1]
    train_y[i] = DATA_Y[n]


svm = SVC()

begin = time.perf_counter_ns()
svm.fit(train_x, train_y)
end = time.perf_counter_ns()
svm_train_time = (end - begin)//1000/1000

begin = time.perf_counter_ns()
svm_result = svm.predict(DATA_X)
end = time.perf_counter_ns()
svm_predict_time = (end - begin)//1000/1000

knn = KNeighborsClassifier()

begin = time.perf_counter_ns()
knn.fit(train_x, train_y)
end = time.perf_counter_ns()
knn_train_time = (end - begin)//1000/1000

begin = time.perf_counter_ns()
knn_result = knn.predict(DATA_X)
end = time.perf_counter_ns()
knn_predict_time = (end - begin)//1000/1000

svm_error=0
knn_error=0
for i in range(DATA_SIZE):
    if svm_result[i] != DATA_Y[i]:
        svm_error+=1
    if knn_result[i] != DATA_Y[i]:
        knn_error+=1
knn_error = math.floor(knn_error/DATA_SIZE*100)
svm_error = math.floor(svm_error/DATA_SIZE*100)


pairs = np.zeros((500*500, 2))
for i in range(0, 500):
    for j in range(0, 500):
        pairs[500*i + j] = [i, j]
svm_contour = svm.predict(pairs)
svm_contour = np.reshape(svm_contour, (500, 500))
svm_contour = np.transpose(svm_contour)
knn_contour = knn.predict(pairs)
knn_contour = np.reshape(knn_contour, (500, 500))
knn_contour = np.transpose(knn_contour)
fig, (svm_graph, knn_graph) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
x = np.arange(0, 500, 1)
y = np.arange(0, 500, 1)
xx, yy = np.meshgrid(x, y)


svm_graph.set_xlabel('SVM classification')
knn_graph.set_xlabel('KNN classification')

svm_graph.annotate('Время обучения ' + str(svm_train_time) + ' ms', xy =(10, 10))
knn_graph.annotate('Время обучения ' + str(knn_train_time) + ' ms', xy =(10, 10))
svm_graph.annotate('Время предсказания ' + str(svm_predict_time) + ' ms', xy =(10, 40))
knn_graph.annotate('Время предсказания ' + str(knn_predict_time) + ' ms', xy =(10, 40))
svm_graph.annotate('Процент ошибок ' + str(svm_error) + '%', xy =(10, 470))
knn_graph.annotate('Процент ошибок ' + str(knn_error) + '%', xy =(10, 470))
for i in range(DATA_SIZE):
    if DATA_Y[i] == 1:
        svm_graph.scatter(DATA_X[i][0], DATA_X[i][1], c='red', edgecolor='black')
    elif DATA_Y[i] == 2:
        svm_graph.scatter(DATA_X[i][0], DATA_X[i][1], c='blue', edgecolor='black')
    elif DATA_Y[i] == 3:
        svm_graph.scatter(DATA_X[i][0], DATA_X[i][1], c='yellow', edgecolor='black')


for i in range(DATA_SIZE):
    if DATA_Y[i] == 1:
        knn_graph.scatter(DATA_X[i][0], DATA_X[i][1], c='red', edgecolor='black')
    elif DATA_Y[i] == 2:
        knn_graph.scatter(DATA_X[i][0], DATA_X[i][1], c='blue', edgecolor='black')
    elif DATA_Y[i] == 3:
        knn_graph.scatter(DATA_X[i][0], DATA_X[i][1], c='yellow', edgecolor='black')
svm_graph.contourf(xx, yy, svm_contour, levels = 3, colors=['red', 'blue', 'yellow', 'yellow'], alpha=0.5)
knn_graph.contourf(xx, yy, knn_contour, levels = 3, colors=['red', 'blue', 'yellow', 'yellow'], alpha=0.5)
print(DATA_Y)
print(svm_result)
plt.show()
