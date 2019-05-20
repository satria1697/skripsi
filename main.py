#Import Library
import collections
import io
import json
import math
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


#Tokenisasi data dan memfilter punctuation dan stopwords
def Tokenisasi(text):
  cleanText = []
  sentences = []
  text = text.lower()
  sent_detector = PunktSentenceTokenizer(lang_vars = LangVars())
  para = text.split('\n\n')
  for paragraf in para:
      sentencess = sent_detector.tokenize(paragraf)
      stopWords = set(stopwords.words('english'))
      for s in sentencess:
        sentences.append(s)
        tokens = word_tokenize(s)
        words = [word for word in tokens if word.isalpha()]
        words = [w for w in words if not w in stopWords]
        sente = " ".join(words)
        cleanText.append(sente)
  return sentences, cleanText

#array 0
def arrayzero(text, jdata, sentencez):
  targetV = []
  sent_detector = PunktSentenceTokenizer(lang_vars = LangVars())
  count = 0
  arrayV = []
  paragraph = text.split("\n\n")
  for i in range(len(sentencez)):
    targetV.append(0)
  for para in paragraph:
    sentences = sent_detector.tokenize(para)
    for k in sentences:
      if k.lower() == sentencez[len(sentencez)-1]:
        count += len(k)
      else:
        count += len(k)+1
        if sentences.index(k) == len(sentences)-1:
          count += 2
      arrayV.append(count)
  # print(arrayV)
  for j in range(len(arrayV)):
    if arrayV[j] in jdata:
      targetV[j] = 1
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
    if nilai[0][1]:
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
    nilai = pos[0][1]/count
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

#Memanggil Data
corpus_root = 'D:\Education\SKRIPSI\pan18-style-change-detection-training-dataset-2018-01-31'
DataCorpus = PlaintextCorpusReader(corpus_root,'.*txt')
DataJson = PlaintextCorpusReader(corpus_root,'.*truth')
fileTest = DataCorpus.open('problem-1.txt')
text = fileTest.read()
fileTest.close()
print(text)
# print(len(text))
class LangVars(PunktLanguageVars):
    sent_end_chars = ('.', '?', '!', '...', '-', '.)', '\n\n')
jsonfile = DataJson.open('problem-1.truth')
jsonstr = jsonfile.read()
jdata = json.loads(jsonstr)['positions']
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

params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(vektorNpS, targetV)

# Plot feature importance
feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.array(['5% percentile', '50% percentile', '95% percentile', 'Punctuation', 'Tag'])
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos)
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()


# mse = mean_squared_error(y_test, clf.predict(X_test))
# print("MSE: %.4f" % mse)
