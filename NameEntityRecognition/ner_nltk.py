# NameEntityRecognition
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import nltk

def name_entity_nltk_1(s):
    words = nltk.word_tokenize(s)
    tagged = nltk.pos_tag(words)
    ne = nltk.ne_chunk(tagged,binary = True)
    return ne

def name_entity_nltk_2(s):
    words = nltk.word_tokenize(s)
    tagged = nltk.pos_tag(words)
    ne = nltk.ne_chunk(tagged,binary = False)
    return ne

print name_entity_nltk_1("My name is Bob and i am working at Google in America.")
print name_entity_nltk_2("My name is Bob and i am working at Google in America.")
