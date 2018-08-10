#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 11:38:20 2017

@author: d7-02
"""
import spacy
nlp = spacy.load('en')

doc = nlp(u'Apple is looking at buying U.K. startup for $1 billion')
'''
for i in test_sen:
    doc = nlp(i.decode('utf8'))
    print i+'\n'
    for ent in doc.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)
    print '\n\n'
'''    
def ner_spacy(s):
    doc = nlp(s.decode('utf-8'))
    l = []
    print s+'\n'
    for ent in doc.ents:
        #print(ent.text, ent.start_char, ent.end_char, ent.label_)
        l1 = [str(ent.text), ent.start_char, ent.end_char, str(ent.label_)]
        l.append(l1)
    return l
        
print ner_spacy('Apple is looking at buying U.K. startup for $1 billion')    
