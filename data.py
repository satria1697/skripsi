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
X = df.iloc[:, 1:31]
y = df.iloc[:, 31:]
X_train = np.array(X)
y_train = np.array(y)

gbr = GradientBoostingRegressor(n_estimators=200, max_depth=4, verbose=0)
gbr.fit(X_train, y_train)

testData = 'D:\Education\SKRIPSI\\PYTHON\\skripsi\\validation\\1-vektor\\3-gram'
DataTest = [i for i in os.listdir(testData) if i.endswith("csv")]
DataTest = natsort.natsorted(DataTest)

jumlahData = len(DataTest)

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

for data in predictionGG[:10]:
  plt.subplot(211)
  plt.plot(range(len(data)), data)
  plt.xlabel('sentence')
  plt.show()