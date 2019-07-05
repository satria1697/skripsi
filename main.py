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
import natsort

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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
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
def arrayzero(text, jdata, sentencez, authorZ):
  sent_detector = PunktSentenceTokenizer(lang_vars = LangVars())
  index = 0
  count = 0
  geser = 1
  countKalimat = 0
  targetV = []
  arrayV = []
  if not jdata:
    targetV = [0] * len(sentencez)
  else:
    if authorZ[0] == '1':
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
    for j in range(len(arrayV)):
      if arrayV[j] in jdata:
        l = j
        if authorZ[geser] == '1':
          while l < len(arrayV):
            # print(str(geser) + ' ' + str(len(authorz)))
            targetV[l] = 0
            l += 1
        else:
          while l < len(arrayV):
            # print(str(geser) + ' ' + str(len(authorz)))
            targetV[l] = 1
            l += 1
        geser += 1
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

#lexical Score
def lexi(cleanText):
  lexiScore = [0]
  kalimatGabung = []
  token = []
  for i in range(len(cleanText)-1):
    kalimatGabung.append(cleanText[i] + ' ' + cleanText[i+1])
  for kata in kalimatGabung:
    token.append(nltk.word_tokenize(kata))
  for i in range(len(token)-1):
    token1 = token[i]
    token2 = token[i+1]
    token1.sort()
    token2.sort()
    i = 0
    lexi = 0
    jToken12 = 0
    jToken22 = 0
    while i < len(token1):
      j = 0
      for j in range(len(token2)):
        jToken1 = token1.count(token1[i])
        jToken2 = token2.count(token2[j])
        lexi += jToken1 * jToken2
        jToken12 += jToken1
        jToken22 += jToken2
        break
      jj = math.sqrt((jToken12**2)*(jToken22**2))
      if jj == 0:
        lexi = 0
      else:
        lexi= lexi / jj
      i += token1.count(token1[i])
    lexiScore.append(lexi)
  lexiScore.append(0)
  return lexiScore

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
def vektorS(percen5, wfa, percen95, counterPuncA, posTagA, lexScore):
  vektorS = []
  for i in range(len(wfa)):
    vektorSa = []
    vektorSa.append(percen5[i])
    vektorSa.append(wfa[i])
    vektorSa.append(percen95[i])
    vektorSa.append(counterPuncA[i])
    vektorSa.append(posTagA[i])
    vektorSa.append(lexScore[i])
    vektorS.append(vektorSa)
  return vektorS

class LangVars(PunktLanguageVars):
  sent_end_chars = ('.', '?', '!', '...', '.)', '\n\n', '\n', '.\n')

corpus_root = 'D:\Education\SKRIPSI\pan18-style-change-detection-training-dataset-2018-01-31'
corpus_root_test = 'D:\Education\SKRIPSI\pan18-style-change-detection-validation-dataset-2018-01-31'
DataCorpusTest = [i for i in os.listdir(corpus_root_test) if i.endswith("txt")]
DataJsonTest = [i for i in os.listdir(corpus_root_test) if i.endswith("truth")]
DataCorpus = [i for i in os.listdir(corpus_root) if i.endswith("txt")]
DataJson = [i for i in os.listdir(corpus_root) if i.endswith("truth")]
DataCorpus = natsort.natsorted(DataCorpus)
DataJson = natsort.natsorted(DataJson)
DataCorpusTest = natsort.natsorted(DataCorpusTest)
DataJsonTest = natsort.natsorted(DataJsonTest)

# metaData = pd.read_csv('train_meta.csv')
metaData = pd.read_csv('validation_meta.csv')
author = metaData.mixMethod
author = author.values
authorZ = []
for penulis in author:
  authorS = penulis.split('-')
  authorZZ = []
  for penu in authorS:
    authorZZ.append(penu.replace('A',''))
  authorZ.append(authorZZ)
# print(authorZ)

#inisialisasi data yang digunakan
jumlahData = len(DataCorpusTest)

#Pembuatan fitur
# sen = []
# target = []
for i in tqdm(range(jumlahData)):
  sen = []
  target = []
  # fileCoba = os.path.join(corpus_root, DataCorpus[i])
  # fileJson = os.path.join(corpus_root, DataJson[i])
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

  authorZe = authorZ[i]

  sentences, cleanText = Tokenisasi(text)
  lexScore = lexi(cleanText)
  counterpunc = punctt(sentences)
  postag = postagg(sentences)

  #ctrl+/ untuk comment atau uncomment
  #1-gram
  # sorted_dict, result = freqWordDoc(cleanText, 1)
  # senResult =  freqWordSent(cleanText, 1)
  # wfa = wordFreq(result, sorted_dict, cleanText, senResult, 1)

  #3-gram
  # sorted_dict, result = freqWordDoc(cleanText, 3)
  # senResult =  freqWordSent(cleanText, 3)
  # wfa = wordFreq(result, sorted_dict, cleanText, senResult, 3)

  #4-gram
  sorted_dict, result = freqWordDoc(cleanText, 4)
  senResult =  freqWordSent(cleanText, 4)
  wfa = wordFreq(result, sorted_dict, cleanText, senResult, 4)

  percen5, percen95 = percentile(wfa)
  meanwfa = meanz(wfa)
  targetV = arrayzero(text, jdata, sentences, authorZe)
  vektorNpS = vektorS(percen5, meanwfa, percen95, counterpunc, postag, lexScore)
  vektorNpS = np.array(vektorNpS)
  # print(vektorNpS)
  # print(targetV)

  for j in range(len(vektorNpS)):
    senS = []
    if j-2<0:
      senS.append([0,0,0,0,0,0])
    else:
      senS.append(vektorNpS[j-2])
    if j-1<0:
      senS.append([0,0,0,0,0,0])
    else:
      senS.append(vektorNpS[j-1])
    senS.append(vektorNpS[j])
    if j+1>len(vektorNpS)-1:
      senS.append([0,0,0,0,0,0])
    else:
      senS.append(vektorNpS[j+1])
    if j+2>len(vektorNpS)-1:
      senS.append([0,0,0,0,0,0])
    else:
      senS.append(vektorNpS[j+2])
    # senS = np.array(senS)
    # print(senS)
    # target = np.append(target, np.full((5,1), targetV[i]))
    # print(target)
    # target = pd.DataFrame(target)
    #   (sen, target.values.ravel())
    senS = np.concatenate(senS, axis=None)
    sen.append(senS)
    target.append([targetV[j]])
  bgone = np.concatenate((sen,target), axis=1)
  # print(bgone)
  df = pd.DataFrame(bgone)
  df.to_csv(r'D:\Education\SKRIPSI\PYTHON\validation\4-gram\problem-'+str(i+1)+'.csv')
  # print(sen)
# bgone = np.concatenate((sen,target), axis=1)
# df = pd.DataFrame(bgone)
# df.to_csv('training-4-gram.csv')