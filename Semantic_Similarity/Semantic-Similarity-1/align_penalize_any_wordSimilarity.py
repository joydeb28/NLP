import nltk
import sys
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
import pandas as pd
import re
import math


ALPHA = 0.2
BETA = 0.45
ETA = 0.4
PHI = 0.2
DELTA = 0.85
 


 
#data = pd.read_csv('java.txt', sep="\n\n",header=None) 



'''
############################################################################
############################################################################
                            Acronym dict
############################################################################
############################################################################  
'''
acronym=[
{'2f4u':'too fast for you'},
{'fyeo':'for your eyes only'},
{'4yeo':'for your eyes only'},
{'aamof':'as a matter of fact'},
{'ack':'acknowledgment'},
{'afaik':'as far as i know'},
{'afair':'as far as i remember'},
{'afair':'as far as i recall'},
{'afk':'away from keyboard'},
{'aka':'also known as'},
{'btk':'back to keyboard'},
{'b2k':'back to keyboard'},
{'btt':'back to topic'},
{'btw':'by the way'},
{'b/c':'because'},
{'c&p':'copy and paste'},
{'cu':'see you'},
{'cys':'check your settings'},
{'diy':'do it yourself'},
{'eobd':'end of business day'},
{'eod':'end of discussion'},
{'eom':'end of message'},
{'eot':'end of thread'},
{'eot':'end of text'},
{'eot':'end of transmission'},
{'faq':'frequently asked questions'},
{'fack':'full acknowledge'},
{'fka':'formerly known as'},
{'fwiw':'for what its worth'},
{'jfyi':'just your information'},
{'fyi' :'for your information'},
{'ftw':'for the win'},
{'ftw':'fuck the world'},
{'hf'	:'have fun'},
{'hth':'hope this helps'},
{'idk':'i dont know'},
{'iirc':'remember correctly'},
{'iirc':'if i recall'},
{'imho':'in my humble opinion'},
{'imo':'in my opinion'},
{'imnsho':'in my not so honest opinion'},
{'imnsho':'in my not so humble opinion'},
{'iow':'in other words'},
{'itt':'in this thread'},
{'lol':'laughing out loud'},
{'dgmw':'dont get me wrong'},
{'mmw':'mark my words'},
{'n/a':'not applicable'},
{'n/a':'not available'},
{'nan':'not a number'},
{'nntr':'no need to reply'},
{'noob':'newbie'},
{'noyb':'none of your business'},
{'nrn':'no reply necessary'},
{'omg':'oh my god'},
{'op':'original post'},
{'op':'original poster'},
{'ot':'off topic'},
{'otoh':'on the other hand'},
{'pebkac':'problem exists between keyboard and chair'},
{'pov':'point of view'},
{'rotfl':'rolling on the floor laughing'},
{'rtfm':'read the fine manual'},
{'scnr':'sorry, could not resist'},
{'sflr':'sorry: for late reply'},
{'spoc':'single point of contact'},
{'tba':'to be announced'},
{'tbc':'to be confirmed'},
{'tbc':'to be continued'},
{'tia':'thanks in advance'},
{'tgif':'thanks god, its friday'},
{'thx':'thanks'},
{'tnx':'thanks'},
{'tq'	:'thank you'},
{'tyvm':'thank you very much'},
{'tyt':'take your time'},
{'ttyl':'talk to you later'},
{'woot':'hooray'},
{'wfm':'works for me'},
{'wrt':'with regard to'},
{'wth':'what the hell'},
{'wth':'what the heck'},
{'wtf':'what the fuck'},
{'ymmd':'you made my day'},
{'ymmv':'your mileage may vary'},
{'yam':'yet another meeting'},
{'icymi':'in case you missed it'},
{"can't" : 'can not'},
{"couldn't" : "could not"},
{"haven't":"have not"},
{"hadn't" : "had not"},
{"hasn't" : "has not"},
{"couldn't":"could not"},
{"aren't":"are not"},
{"don't" : "do not"},
{"doesn't" : "does not"},
{"didn't" : "did not"},
{"i.e." : "that is"},
{"e.g." : "for example"}
]

'''
############################################################################
############################################################################
                            Number to Words
############################################################################
############################################################################  
'''
def numToWords(num,join=True):
    #words = {} convert an integer number into words
    units = ['','one','two','three','four','five','six','seven','eight','nine']
    teens = ['','eleven','twelve','thirteen','fourteen','fifteen','sixteen', \
             'seventeen','eighteen','nineteen']
    tens = ['','ten','twenty','thirty','forty','fifty','sixty','seventy', \
            'eighty','ninety']
    thousands = ['','thousand','million','billion','trillion','quadrillion', \
                 'quintillion','sextillion','septillion','octillion', \
                 'nonillion','decillion','undecillion','duodecillion', \
                 'tredecillion','quattuordecillion','sexdecillion', \
                 'septendecillion','octodecillion','novemdecillion', \
                 'vigintillion']
    words = []
    if num==0: words.append('zero')
    else:
        numStr = '%d'%num
        numStrLen = len(numStr)
        groups = (numStrLen+2)/3
        numStr = numStr.zfill(groups*3)
        for i in range(0,groups*3,3):
            h,t,u = int(numStr[i]),int(numStr[i+1]),int(numStr[i+2])
            g = groups-(i/3+1)
            if h>=1:
                words.append(units[h])
                words.append('hundred')
            if t>1:
                words.append(tens[t])
                if u>=1: words.append(units[u])
            elif t==1:
                if u>=1: words.append(teens[u])
                else: words.append(tens[t])
            else:
                if u>=1: words.append(units[u])
            if (g>=1) and ((h+t+u)>0): words.append(thousands[g]+',')
    if join: return ' '.join(words)
    return words

'''
############################################################################
############################################################################
                            Preprocessing
############################################################################
############################################################################  
'''
to_be_removed = ".!?;,:()"
file_corpus = open("java.txt","r+")
data = file_corpus.read()
corpus = data.split('.')
for line in range(len(corpus)):
    words=[]
    sen = corpus[line]
    for word in sen.split():
        flag=0
        if re.match(r'^\d+$',word):
            num = int(float(word))
            new_sen = numToWords(num)
            for c in to_be_removed:
                new_sen = new_sen.replace(c, '')
            new_words = new_sen.split()
            for i in new_words:
                words.append(i)
        if re.match(r"\w+[-]\w+", word):
            multi_words = word.split("-")
            for i in multi_words:
                words.append(i)
            flag = 1
        #print word
        if flag!=1:
            words.append(word)
        #print words
    corpus[line] = ' '.join(i.lower() for i in words)
all_words=[]
all_words = [word for line in corpus for word in line.split()]
all_words.sort()
"""
############################################################################
############################################################################
                            POS LIST
############################################################################
############################################################################        
"""
p1=['NN','NNS','NNP','NNPS','VB','VBD','VBG','VBN','VBP','VBZ','PRP','PRP$','CD']
"""
############################################################################
############################################################################
                            word similarity
############################################################################
############################################################################        
"""
    
def Word_similarity(word_1, word_2):
    '''
    add any word similarity as er your need from wordnet or make your own
    '''
        
"""
############################################################################
############################################################################
                            word' similarity
############################################################################
############################################################################        
"""
subject_pronouns = ['I','he','she','we','they']
object_pronouns = ['me','him','her','us','them']
def word_sim_dash(w1,w2):
    #if both are same
    if(w1 == w2):
        return 1
    # subject pronouns & object pronouns
    if(w1 in subject_pronouns and w2 in object_pronouns):
        for i in len(subject_pronouns):
            if(subject_pronouns[i]==w1):
                if(object_pronouns[i]==w2):
                    return 1
    if(w2 in subject_pronouns and w1 in object_pronouns):
        for i in len(subject_pronouns):
            if(subject_pronouns[i]==w2):
                if(object_pronouns[i]==w1):
                    return 1
    #acronym
    if(w1 in acronym):
        w_list = acronym[w1].split(" ")
        if(w2 == w_list[0] or w2==w1):
            return 1
    return 0

"""
############################################################################
############################################################################
                Best pairs with similarity score
############################################################################
############################################################################        
"""
def makeList(s1,s2):
    l = []
    w1 = word_tokenize(s1)
    w2 = word_tokenize(s2)
    pos1 = nltk.pos_tag(w1)
    for i in pos1:
        mx = 0
        t = i[0]
        gt = ''
        ps = ''
        for j in w2:
            v = Word_similarity(t,j)
            if mx < v:
                mx = v
                gt = j
                ps = i[1]
        l.append([t,gt,mx,ps])
    #print l
    return l
"""
############################################################################
############################################################################
                            AntonymsCheck between two words
############################################################################
############################################################################        
"""
def antonym_check(w1,w2):
    antonyms = []
    for syn in wn.synsets(w1):
        for l in syn.lemmas():
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())
    for syn2 in wn.synsets(w2):
        for m in syn2.lemmas():
            for n in antonyms:
                if m.name()==n:
                    return 1
"""
############################################################################
############################################################################
                Penality for less similarity(>0.05)
############################################################################
############################################################################        
"""
def penalty_less_similar_words(s1,s2):
    l1 = makeList(s1,s2)
    a = []
    for i in l1:
        a.append(i)
        if(Word_similarity(i[0],i[1])>0.05):
            a.remove(i)    
    r=0            
    for i in a:
        if i[3] in p1: 
            r=r + i[2] + (frequency_count(i[0])*1)
        else:
            r=r + i[2]+(frequency_count(i[0])*0.5) 
    return r               
"""
############################################################################
############################################################################
                            Penalty for opposite words
############################################################################
############################################################################        
"""    
def penalty_antonym_words(s1,s2):
    l = makeList(s1,s2)
    penalty_count = 0
    for i in l:
        if antonym_check(i[0],i[1])==1:
            penalty_count+=1
    return penalty_count*0.5
"""
############################################################################
############################################################################
                            Frequency Count
############################################################################
############################################################################        
"""   
def frequency_count(word):
    N=1
    if word in all_words:
        N = N + 1
    if N>1:
        return 1/math.log(N)
    else:
        return 1   
"""
############################################################################
############################################################################
                            Similarity Score of two sentence
############################################################################
############################################################################        
"""    
def total_score(s1,s2):
    l1=makeList(s1,s2)
    l2=makeList(s2,s1)
    total_score_sen1 = 0
    total_score_sen2 = 0
    for i in l1:
        total_score_sen1 = total_score_sen1+i[2]
        #print total_score_sen1
    total_score_sen1=total_score_sen1/(2*len(s1))
    #print (total_score_sen1)
    penalty_sen1 = penalty_less_similar_words(s1,s2)
    penalty_sen1 = penalty_sen1/(2*len(s1))

    penalty_antonyms_sen1 = penalty_antonym_words(s1,s2)
    penalty_antonyms_sen1 = penalty_antonyms_sen1/(2*len(s1))

    for i in l2:
        total_score_sen2=total_score_sen2+i[2]
    #print (total_score_sen2)
   
    total_score_sen2=total_score_sen2/(2*len(s2))
    penalty_sen2 = penalty_less_similar_words(s2,s1)
    penalty_sen2 = penalty_sen2/(2*len(s2))

    penalty_antonyms_sen2 = penalty_antonym_words(s2,s1)
    penalty_antonyms_sen2 = penalty_antonyms_sen2/(2*len(s2))

    penalty = penalty_sen1 + penalty_sen2 + penalty_antonyms_sen1 + penalty_antonyms_sen2
   
    total_score = total_score_sen1 + total_score_sen2 - penalty
    #print total_score_sen1
    return total_score

#print (total_score("sentence 1","sentence 2"))
