import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import os
import natsort
from tqdm import tqdm

df = pd.read_csv('training-3-gram.csv')
X = df.iloc[:, 0:30]
y = df.iloc[:, 30:]
X_train = np.array(X)
y_train = np.array(y)
# print(X)
# print(y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

testData = 'D:\Education\SKRIPSI\\PYTHON\\validation\\3-gram'
DataTest = [i for i in os.listdir(testData) if i.endswith("csv")]
DataTest = natsort.natsorted(DataTest)

jumlahData = len(DataTest)

gbr = GradientBoostingRegressor(n_estimators=200, max_depth=4, verbose=0)
gbr.fit(X_train, y_train)

predictionGG = []
for i in tqdm(range(jumlahData)):
  csvfile = os.path.join(testData,DataTest[i])
  df = pd.read_csv(csvfile)
  X = df.iloc[:, 0:30]
  y = df.iloc[:, 30:]
  X_test = np.array(X)
  y_test = np.array(y)
  predictionG = []
  for data in X_test:
    data = data.reshape(1,-1)
    y_pred = gbr.predict(data)
    predictionG.append(y_pred)
    predictionGG.append(predictionG)
  # np.asarray(predictionG)
  # plt.subplot(211)
  # plt.plot(range(len(predictionG)), predictionG)
  # plt.xlabel('sentence')
  # plt.show()

np.asarray(predictionGG)
plt.subplot(211)
plt.plot(range(len(predictionG)), predictionG)
plt.xlabel('sentence')
plt.show()

# lr = LogisticRegression(verbose=2)
# lr.fit(X_train, y_train)
# predictionL = []
# for data in X_test:
#   data = data.reshape(1,-1)
#   y_pred = lr.predict(data)
#   predictionL.append(y_pred)
# print(predictionL)

# sv = svm.SVC(kernel='linear', verbose=1, gamma='scale')
# sv.fit(X_train, y_train)
# print(sv.predict(X_test))