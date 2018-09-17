# Part-of-Speech-Tagging

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 13:30:28 2017

@author: d7-02
"""

import nlpnet
nlpnet.set_data_dir('dependency')

#parser = nlpnet.DependencyParser('dependency', language='en')
#tagger = nlpnet.POSTagger('/path/to/pos-model/', language='pt')

tagger = nlpnet.POSTagger()
print tagger.tag(u"I want to book a flight from Delhi to Pune on Sunday")
#parsed_text = parser.parse(u'I want to book a flight from Delhi to Pune on Sunday')
#sent = parsed_text[0]
#print(sent.to_conll())
