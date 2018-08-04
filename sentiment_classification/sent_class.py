import nltk.corpus
from nltk.corpus import nps_chat
from nltk.tokenize import TweetTokenizer
 
class QuestionDetector():
 
    def __init__(self):
        posts = nltk.corpus.nps_chat.xml_posts()
        featuresets = [(self.__dialogue_act_features(post.text), post.get('class')) for post in posts]
        size = int(len(featuresets) * 0.1)
        train_set, test_set = featuresets[size:], featuresets[:size]
        self.classifier = nltk.NaiveBayesClassifier.train(train_set)
        Question_Words = ['what', 'where', 'when','how','why','did','do','does','have',
                          'has','am','is','are','can','could','may','would','will','should',
                          "didn't","doesn't","haven't","isn't","aren't","can't","couldn't",
                          "wouldn't","won't","shouldn't",'?']
        self.Question_Words_Set = set(Question_Words)
        self.tknzr = TweetTokenizer()

    def __dialogue_act_features(self,sentence):
         features = {}
         for word in nltk.word_tokenize(sentence):
             features['contains({})'.format(word.lower())] = True
         return features
    
    def IsQuestion(self,sentence):
        if "?" in sentence:
            return True
        tokens = self.tknzr.tokenize(sentence.lower())
        if self.Question_Words_Set.intersection(tokens) == False:
            return False
        predicted = self.classifier.classify(self.__dialogue_act_features(sentence))
        if predicted == 'whQuestion' or predicted == 'ynQuestion':
            return True
         
        return False

def test(sent):
    qo = QuestionDetector()
    return qo.IsQuestion(sent)
    