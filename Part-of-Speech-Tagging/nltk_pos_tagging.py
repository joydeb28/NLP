# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 17:59:13 2017

@author: d7-02
"""
import nltk

s = 'I want to book a flight from Delhi to Pune on Sunday.'
w = nltk.word_tokenize(s)
sentence = nltk.pos_tag(w)
print(sentence)
'''
grammar = "NP: {<DT>?<JJ>*<NN>}"

cp = nltk.RegexpParser(grammar)
result = cp.parse(sentence)
print(result)
result.draw()
'''
