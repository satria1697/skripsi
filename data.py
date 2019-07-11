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
from tqdm import trange

for i in trange(1,4):
  for j in trange(2,5):
    df = pd.read_csv('training-' + str(i) + '-vektor-' + str(j) + '-gram.csv')
    if i == 1:
      X = df.iloc[:, 1:18]
      y = df.iloc[:, 18]
    if i == 2:
      X = df.iloc[:, 1:31]
      y = df.iloc[:, 31]
    if i == 3:
      X = df.iloc[:, 1:42]
      y = df.iloc[:, 42]
    X_train = np.array(X)
    y_train = np.array(y)

    gbr = GradientBoostingRegressor(n_estimators=200, max_depth=4, verbose=0)
    gbr.fit(X_train, y_train)
    
    testData = 'D:\Education\SKRIPSI\\PYTHON\\skripsi\\validation\\' + str(i) +'-vektor\\' + str(j) + '-gram'
    DataTest = [f for f in os.listdir(testData) if f.endswith("csv")]
    DataTest = natsort.natsorted(DataTest)

    jumlahData = len(DataTest)

    predictionGG = []
    for g in trange(jumlahData):
      csvfile = os.path.join(testData,DataTest[g])
      df = pd.read_csv(csvfile)
      if i == 1:
        X = df.iloc[:, 1:18]
        y = df.iloc[:, 18]
      if i == 2:
        X = df.iloc[:, 1:31]
        y = df.iloc[:, 31]
      if i == 3:
        X = df.iloc[:, 1:42]
        y = df.iloc[:, 42]

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

    pos = 0
    for data in predictionGG:
      pos += 1
      plt.subplot(211)
      plt.plot(range(len(data)), data)
      plt.title('problem-' + str(pos))
      plt.xlabel('sentence')
      plt.savefig('D:\Education\SKRIPSI\\figur\\' + str(i) +'-vektor\\' + str(j) + '-gram\\problem' + str(pos))
      plt.close()
      # plt.show()