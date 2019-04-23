#Import Library
import collections
import io
import math
import re
import string
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import numpy as np
from nltk import ngrams
from nltk.corpus import PlaintextCorpusReader, stopwords
from nltk.probability import FreqDist
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize


#Tokenisasi data dan memfilter punctuation dan stopwords
def Tokenisasi(text):
  cleanText = []
  text = text.lower()
  sentences = sent_tokenize(text)
  stopWords = set(stopwords.words('english'))
  # print(sentences)
  for s in sentences:
    tokens = word_tokenize(s)
    words = [word for word in tokens if word.isalpha()]
    words = [w for w in words if not w in stopWords]
    sente = " ".join(words)
    # print(sente)
    cleanText.append(sente)
  # print(cleanText)
  return sentences, cleanText
  # porter = PorterStemmer()
  # stemmed = [porter.stem(word) for word in tokens]
  # print(stemmed)


#Membuat Vocab
def vocab(cleanText):
  firstEncounter = []
  for s in cleanText:
    cleanWord = word_tokenize(s)
    for w in cleanWord:
      if w not in firstEncounter:
        firstEncounter.append(w)
  firstEncounter.sort()
  # print(firstEncounter)
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
  # print(sorted_dict)
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
  # print(sResult)
  return senResult
  
#Word Frequency
def wordFreq(result, sorted_dict, cleanText, senResult, n):
  nd = list(sorted_dict.values())[0]
  # print(nd)
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
      # print(s,word)
      nsw = senResult[s][word]
      # print(nsw)
      ndw = result.get(word)
      # print(ndw)
      vsw = math.log((nd/(ndw-nsw+1)), 2)
      wfs[word] = vsw
      wfas.append(vsw)
    wf[counter] = wfs
    counter+=1
    wfa.append(wfas)
  # print(wf)
  # print(wfa)
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
  # print(meanWordFreq)
  return meanWordFreq

#percentile
def percentile(wfa):
  percen5 = []
  percen95 = []
  for value in wfa:
    value.sort()
    # print(value)
    percen5.append(np.percentile(value, 5))
    percen95.append(np.percentile(value, 95))
  return percen5, percen95

#Punctuation
def punctt(sentences):
  counterPunc = {}
  counterPuncA = []
  counter = 1
  punc = set(string.punctuation)
  for s in sentences:
    cleanWord = word_tokenize(s)
    for word in cleanWord:
      cp = {}
      cpas = []
      for char in word:
        if char in punc:
          if char not in counterPunc:
            cp[char] = 0
          cp[char] += 1
    counterPunc[counter] = cp
    counter += 1
    count = len(re.findall(r'\w+', s))
    nilai = FreqDist(cp).most_common(1)
    counterPuncA.append(nilai[0][1])
  return counterPuncA

#Part-of-speech Counter
def postagg(sentences):
  # posTag = {}
  posTagA = []
  counter = 1
  for s in sentences:
    cleanWord = word_tokenize(s)
    posT = nltk.pos_tag(cleanWord)
    count = len(re.findall(r'\w+', s))
    postTagFd = nltk.FreqDist(tag for (word, tag) in posT)
    # posTag[counter] = postTagFd.most_common(1)
    pos = postTagFd.most_common(1)
    nilai = pos[0][1]/count
    posTagA.append(nilai)
    counter += 1
  # print(posTagA)
  return posTagA

#Membuat Vektor
def vektorS(percen5, wfa, percen95, counterPuncA, posTagA):
  vektorS = []
  for i in range(len(wfa)):
    vektorSa = []
    for j in range(5):
      vektorSa.append(percen5[i])
      vektorSa.append(wfa[i])
      vektorSa.append(percen95[i])
      vektorSa.append(counterPuncA[i])
      vektorSa.append(posTagA[i])
    vektorS.append(vektorSa)
  print(vektorS)

#Memanggil Data
corpus_root = 'D:\Education\SKRIPSI\pan18-style-change-detection-training-dataset-2018-01-31'
DataCorpus = PlaintextCorpusReader(corpus_root,'.*txt')
fileTest = DataCorpus.open('problem-1.txt')
text = fileTest.read()
fileTest.close()
print(text)
# fileTest = DataCorpus.raw('problem-1.txt')
# fileTest = fileTest.lower()
# fileTest
sentences, cleanText = Tokenisasi(text)
counterpunc = punctt(sentences)
postag = postagg(sentences)
#ctrl+/ untuk comment atau uncomment
# sorted_dict, result = freqWordDoc(cleanText, 1)
# senResult =  freqWordSent(cleanText, 1)
# wfa = wordFreq(result, sorted_dict, cleanText, senResult, 1)
sorted_dict, result = freqWordDoc(cleanText, 3)
senResult =  freqWordSent(cleanText, 3)
wfa = wordFreq(result, sorted_dict, cleanText, senResult, 3)
# sorted_dict, result = freqWordDoc(cleanText, 4)
# senResult =  freqWordSent(cleanText, 4)
# wfa = wordFreq(result, sorted_dict, cleanText, senResult, 4)
percen5, percen95 = percentile(wfa)
meanwfa = meanz(wfa)
vektorS(percen5, meanwfa, percen95, counterpunc, postag)
