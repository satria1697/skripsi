#Import Library
import collections
import io
import itertools
import json
import math
import os
import re
import string
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import nltk.data
import numpy as np
import pandas as pd
from nltk import ngrams
from nltk.corpus import PlaintextCorpusReader, stopwords
from nltk.probability import FreqDist
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize.punkt import PunktLanguageVars, PunktSentenceTokenizer
from sklearn import datasets, ensemble
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

#Tokenisasi data dan memfilter punctuation dan stopwords
def Tokenisasi(text):
  cleanText = []
  sentences = []
  sent_detector = PunktSentenceTokenizer(lang_vars = LangVars())
  para = text.split('\n\n')
  for paragraf in para:
      sentencess = sent_detector.tokenize(paragraf)
      stopWords = set(stopwords.words('english'))
      for s in sentencess:
        sentences.append(s)
        s = s.lower()
        tokens = word_tokenize(s)
        words = [word for word in tokens if word.isalpha()]
        words = [w for w in words if not w in stopWords]
        sente = " ".join(words)
        cleanText.append(sente)
  # print(sentences)
  return sentences, cleanText

#array 0
def arrayzero(text, jdata, sentencez):
  sent_detector = PunktSentenceTokenizer(lang_vars = LangVars())
  index = 0
  count = 0

  countKalimat = 0
  targetV = []
  arrayV = []
  if len(jdata) == 0:
    targetV = [0] * len(sentencez)
  else:
    targetV = [1] * len(sentencez)
    # print(targetV)
    # print(len(sentencez))
    paragraph = text.split("\n\n")
    for para in paragraph:
      sentences = sent_detector.tokenize(para)
      for k in sentences:
        countKalimat += 1
        if countKalimat == len(sentencez):
          count += len(k)
        else:
          count += len(k)+1
          if sentences.index(k) == len(sentences)-1:
            count += 2
        # print(str(k) + '=' + str(sentencez[index]))
        index+=1
        arrayV.append(count)
    # print(arrayV)
    countPos = 2
    for j in range(len(arrayV)):
      if arrayV[j] in jdata:
        l = j
        while l < len(arrayV):
          # print(str(l) + ' ' + str(len(arrayV)))
          targetV[l] = countPos
          l += 1
        countPos += 1
    # print(targetV)
  return targetV

#Membuat Vocab
def vocab(cleanText):
  firstEncounter = []
  for s in cleanText:
    cleanWord = word_tokenize(s)
    for w in cleanWord:
      if w not in firstEncounter:
        firstEncounter.append(w)
  firstEncounter.sort()
  return 

#Frekuensi kata
#Frekuensi kata terbanyak di dalam 1 dokumen
def freqWordDoc(cleanText, n):
  result = {}
  for s in cleanText:
    gramWordS = []
    gramWord = list(ngrams(s.split(), n, pad_right=True, pad_left=True, right_pad_symbol="", left_pad_symbol=""))
    for word in gramWord:
      joinWord = " ".join(word)
      gramWordS.append(joinWord)
    for word in gramWordS:
      if word in result:
        result[word] += 1 #adding result in the dictionary
      else:
        result[word] = 1
  sorted_result = sorted(result.items(), key=lambda kv: kv[1], reverse=True)
  sorted_dict = collections.OrderedDict(sorted_result)
  return sorted_dict, result

#Frekuensi kata dalam setiap kalimat
def freqWordSent(cleanText, n):
  sResult = {}
  senResult = {}
  counter = 1
  for s in cleanText:
    gramWordS = []
    gramWord = list(ngrams(s.split(), n, pad_right=True, pad_left=True, right_pad_symbol="", left_pad_symbol=""))
    for word in gramWord:
      joinWord = " ".join(word)
      gramWordS.append(joinWord)
    resultz = {}
    for word in gramWordS:
      resultz[word] = word.count(word)
    sResult[counter] = resultz
    counter += 1
    senResult[s] = resultz
  return senResult
  
#Word Frequency
def wordFreq(result, sorted_dict, cleanText, senResult, n):
  nd = list(sorted_dict.values())[0]
  wf = {}
  wfa = []
  counter = 1
  for s in cleanText:
    gramWordS = []
    gramWord = list(ngrams(s.split(), n, pad_right=True, pad_left=True, right_pad_symbol="", left_pad_symbol=""))
    for word in gramWord:
      joinWord = " ".join(word)
      gramWordS.append(joinWord)
    wfs = {}
    wfas = []
    for word in gramWordS:
      nsw = senResult[s][word]
      ndw = result.get(word)
      vsw = math.log((nd/(ndw-nsw+1)), 2)
      wfs[word] = vsw
      wfas.append(vsw)
    wf[counter] = wfs
    counter+=1
    wfa.append(wfas)
  return wfa

#Mean Word Freq
def meanz(wfa):
  meanWordFreq = []
  for value in wfa:
    if len(value):
      nilai = sum(value)/len(value)
    else:
      nilai = 0
    meanWordFreq.append(nilai)
  return meanWordFreq

#percentile
def percentile(wfa):
  percen5 = []
  percen95 = []
  for value in wfa:
    value.sort()
    percen5.append(np.percentile(value, 5))
    percen95.append(np.percentile(value, 95))
  return percen5, percen95

#Punctuation
def punctt(sentences):
  counterPunc = {}
  counterPuncA = []
  counter = 0
  punc = set(string.punctuation)
  for s in sentences:
    cleanWord = word_tokenize(s)
    for word in cleanWord:
      cp = {}
      for char in word:
        if char in punc:
          if char not in counterPunc:
            cp[char] = 0
          cp[char] += 1
    counterPunc[counter] = cp
    counter += 1
    nilai = FreqDist(cp).most_common(1)
    if nilai:
      counterPuncA.append(nilai[0][1])
    else:
      counterPuncA.append(0)
  return counterPuncA

#Part-of-speech Counter
def postagg(sentences):
  posTagA = []
  counter = 1
  for s in sentences:
    cleanWord = word_tokenize(s)
    posT = nltk.pos_tag(cleanWord)
    count = len(re.findall(r'\w+', s))
    postTagFd = nltk.FreqDist(tag for (word, tag) in posT)
    pos = postTagFd.most_common(1)
    if count:
      nilai = pos[0][1]/count
    else:
      nilai = 0
    posTagA.append(nilai)
    counter += 1
  return posTagA

#Membuat Vektor
def vektorS(percen5, wfa, percen95, counterPuncA, posTagA):
  vektorS = []
  for i in range(len(wfa)):
    vektorSa = []
    vektorSa.append(percen5[i])
    vektorSa.append(wfa[i])
    vektorSa.append(percen95[i])
    vektorSa.append(counterPuncA[i])
    vektorSa.append(posTagA[i])
    vektorS.append(vektorSa)
  return vektorS

class LangVars(PunktLanguageVars):
    sent_end_chars = ('.', '?', '!', '...', '.)', '\n\n', '\n', '.\n')

corpus_root = 'D:\Education\SKRIPSI\pan18-style-change-detection-training-dataset-2018-01-31'
corpus_root_test = 'D:\Education\SKRIPSI\pan18-style-change-detection-test-dataset-2018-01-31'
DataCorpusTest = [i for i in os.listdir(corpus_root) if i.endswith("txt")]
DataJsonTest = [i for i in os.listdir(corpus_root) if i.endswith("truth")]
DataCorpus = [i for i in os.listdir(corpus_root) if i.endswith("txt")]
DataJson = [i for i in os.listdir(corpus_root) if i.endswith("truth")]

#inisialisasi data yang digunakan
jumlahDataTraining = 100
jumlahDataTest = 50

#Pembuatan fitur Training
sen = []
target = []
for i in tqdm(range(jumlahDataTraining)):
  fileCoba = os.path.join(corpus_root, DataCorpus[i])
  fileJson = os.path.join(corpus_root, DataJson[i])
  fileTest = open(fileCoba, encoding="utf8")
  text = fileTest.read()
  fileTest.close()
  # print(text)
  # print(len(text))
  # print(DataCorpus[i] + ' ' + DataJson[i])
  jsonfile = open(fileJson)
  jsonstr = jsonfile.read()
  jdata = json.loads(jsonstr)['positions']
  jsonfile.close()
  sentences, cleanText = Tokenisasi(text)
  targetV = arrayzero(text, jdata, sentences)
  counterpunc = punctt(sentences)
  postag = postagg(sentences)
  #ctrl+/ untuk comment atau uncomment
  #1-gram
  # sorted_dict, result = freqWordDoc(cleanText, 1)
  # senResult =  freqWordSent(cleanText, 1)
  # wfa = wordFreq(result, sorted_dict, cleanText, senResult, 1)

  #3-gram
  sorted_dict, result = freqWordDoc(cleanText, 3)
  senResult =  freqWordSent(cleanText, 3)
  wfa = wordFreq(result, sorted_dict, cleanText, senResult, 3)

  #4-gram
  # sorted_dict, result = freqWordDoc(cleanText, 4)
  # senResult =  freqWordSent(cleanText, 4)
  # wfa = wordFre q(result, sorted_dict, cleanText, senResult, 4)

  percen5, percen95 = percentile(wfa)
  meanwfa = meanz(wfa)
  vektorNpS = vektorS(percen5, meanwfa, percen95, counterpunc, postag)
  vektorNpS = np.array(vektorNpS)
  # print(vektorNpS)
  # print(targetV)

  for i in range(len(vektorNpS)):
    senS = []
    if i-2<0:
      senS.append([0,0,0,0,0])
    else:
      senS.append(vektorNpS[i-2])
    if i-1<0:
      senS.append([0,0,0,0,0])
    else:
      senS.append(vektorNpS[i-1])
    senS.append(vektorNpS[i])
    if i+1>len(vektorNpS)-1:
      senS.append([0,0,0,0,0])
    else:
      senS.append(vektorNpS[i+1])
    if i+2>len(vektorNpS)-1:
      senS.append([0,0,0,0,0])
    else:
      senS.append(vektorNpS[i+2])
    # senS = np.array(senS)
    # print(senS)
    # target = np.append(target, np.full((5,1), targetV[i]))
    # print(target)
    # target = pd.DataFrame(target)
    # clf.fit(sen, target.values.ravel())
    senS = np.mean(senS, axis=0)
    sen.append(senS)
    target.append([targetV[i]])

#Pembuatan fitur test
senTest = []
targetTest = []
for i in tqdm(range(jumlahDataTest)):
  fileCoba = os.path.join(corpus_root_test, DataCorpusTest[i])
  fileJson = os.path.join(corpus_root_test, DataJsonTest[i])
  fileTest = open(fileCoba, encoding="utf8")
  text = fileTest.read()
  fileTest.close()
  # print(text)
  # print(len(text))
  # print(DataCorpus[i] + ' ' + DataJson[i])
  jsonfile = open(fileJson)
  jsonstr = jsonfile.read()
  jdata = json.loads(jsonstr)['positions']
  jsonfile.close()
  sentences, cleanText = Tokenisasi(text)
  targetV = arrayzero(text, jdata, sentences)
  counterpunc = punctt(sentences)
  postag = postagg(sentences)
  #ctrl+/ untuk comment atau uncomment
  #1-gram
  # sorted_dict, result = freqWordDoc(cleanText, 1)
  # senResult =  freqWordSent(cleanText, 1)
  # wfa = wordFreq(result, sorted_dict, cleanText, senResult, 1)

  #3-gram
  sorted_dict, result = freqWordDoc(cleanText, 3)
  senResult =  freqWordSent(cleanText, 3)
  wfa = wordFreq(result, sorted_dict, cleanText, senResult, 3)

  #4-gram
  # sorted_dict, result = freqWordDoc(cleanText, 4)
  # senResult =  freqWordSent(cleanText, 4)
  # wfa = wordFre q(result, sorted_dict, cleanText, senResult, 4)

  percen5, percen95 = percentile(wfa)
  meanwfa = meanz(wfa)
  vektorNpS = vektorS(percen5, meanwfa, percen95, counterpunc, postag)
  vektorNpS = np.array(vektorNpS)
  # print(vektorNpS)
  # print(targetV)

  for i in range(len(vektorNpS)):
    senTestS = []
    if i-2<0:
      senTestS.append([0,0,0,0,0])
    else:
      senTestS.append(vektorNpS[i-2])
    if i-1<0:
      senTestS.append([0,0,0,0,0])
    else:
      senTestS.append(vektorNpS[i-1])
    senTestS.append(vektorNpS[i])
    if i+1>len(vektorNpS)-1:
      senTestS.append([0,0,0,0,0])
    else:
      senTestS.append(vektorNpS[i+1])
    if i+2>len(vektorNpS)-1:
      senTestS.append([0,0,0,0,0])
    else:
      senTestS.append(vektorNpS[i+2])
    # sen = np.array(sen)
    # print(sen)
    # target = np.append(target, np.full((5,1), targetV[i]))
    # print(target)
    # target = pd.DataFrame(target)
    # clf.fit(sen, target.values.ravel())
    senTestS = np.mean(senTestS, axis=0)
    senTest.append(senTestS)
    targetTest.append([targetV[i]])

params = {'n_estimators': 200, 'max_depth': 4}

X_train = np.array(sen)
y_train = np.array(target)
# print(X_train)
# print(y_train)
y_train = pd.DataFrame(y_train).values.ravel()
X_test = np.array(senTest)
y_test = np.array(targetTest)
# print(X_train)
# print(y_test)
y_test = pd.DataFrame(y_test).values.ravel()

print('Gradient Boosting Regressor')
clf = ensemble.GradientBoostingRegressor(**params)
clf.fit(X_train, y_train)
# Plot feature importance
feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
tagging = np.array(['5% percentile', '50% percentile', '95% percentile', 'Punctuation', 'Tag'])
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, tagging[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()
mse = mean_squared_error(y_test, clf.predict(X_test))
print('MSE: %.4f' % mse)
print('Score: %.4f' % clf.score(X_test, y_test))
print('-------------------------------------------')
print('\nRandom Forest Regressor')
regressor = RandomForestRegressor(**params)
regressor.fit(X_train, y_train)
# Plot feature importance
feature_importance = regressor.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
tagging = np.array(['5% percentile', '50% percentile', '95% percentile', 'Punctuation', 'Tag'])
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, tagging[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()
mse = mean_squared_error(y_test, regressor.predict(X_test))
print('MSE: %.4f' % mse)
print('Score: %.4f' % regressor.score(X_test, y_test))