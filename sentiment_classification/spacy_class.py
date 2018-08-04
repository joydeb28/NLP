from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.metrics import accuracy_score 
from sklearn.base import TransformerMixin 
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import string
punctuations = string.punctuation

import spacy
spacy.load('en_core_web_sm')
from spacy.lang.en import English

parser = English()


#Custom transformer using spaCy 
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]
    def fit(self, X, y=None, **fit_params):
        return self
    def get_params(self, deep=True):
        return {}

# Basic utility function to clean the text 
def clean_text(text):     
    return text.strip().lower()


#Create spacy tokenizer that parses a sentence and generates tokens
#these can also be replaced by word vectors 
def spacy_tokenizer(sentence):
    tokens = parser(sentence)
    tokens = [tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_ for tok in tokens]
    tokens = [tok for tok in tokens if (tok not in stopwords and tok not in punctuations)]     
    return tokens

#create vectorizer object to generate feature vectors, we will use custom spacy's tokenizer
vectorizer = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1)) 
classifier = LinearSVC()


# Create the  pipeline to clean, tokenize, vectorize, and classify 
pipe = Pipeline([("cleaner", predictors()),
                 ('vectorizer', vectorizer),
                 ('classifier', classifier)])

# Load sample data
train = [
    ('What is the price.', 'YES'),          
    ('is it possible?', 'YES'),
    ('it is possible?', 'NO'),
    ("what is the weather like"),
    ("where are we today"),
    ("why did you do that"),
    ("where is the dog"),
    ("when are we going to leave"),
    ("why do you hate me"),
    ("what is the Answer to question 8"),
    ("what is a dinosour","YES"), 
    ("what do i do in an hour","YES"),
    ("why do we have to leave at 6.00","YES"), 
    ("When is the apointment","YES"),
    ("where did you go","YES"),
    ("why did you do that","YES"),
    ("how did he win","YES"),
    ("why won't you help me","YES"),
    ("when did he find you","YES"),
    ("how do you get it","YES"),
    ("who does all the shipping","YES"),
    ("where do you buy stuff","YES"),
    ("why don't you just find it in the target","YES"),
    ("why don't you buy stuff at target","YES"),
    ("where did you say it was","YES"),
    ("when did he grab the phone","YES"),
    ("what happened at seven am","YES"),
    ("did you take my phone","YES"),
    ("do you like me","YES"),
    ("do you know what happened yesterday","YES"),
    ("did it break when it dropped","YES"),
    ("does it hurt everyday","YES"),
    ("does the car break down often","YES"),
    ("can you drive me home","YES"),
    ("where did you find me","YES"),
    ("can it fly from here to target","YES"),
    ("could you find it for me","YES")
    ] 

test =   [
    'is it ok?',
    "wahat is the cost",
    "can you make it 2000",
    "can you chnage my destination",
    "will you able to make it early",
    "i want to book a flight"
]

# Create model and measure accuracy
pipe.fit([x[0] for x in train], [x[1] for x in train]) 
pred_data = pipe.predict([x for x in test]) 
for (sample, pred) in zip(test, pred_data):
    print sample, pred 
