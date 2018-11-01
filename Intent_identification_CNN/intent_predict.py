from preprocessor import pad_vec_sequences, labels
from keras.models import load_model
import spacy
from preprocessor import nlp
import numpy as np
from dependency_tree import to_nltk_tree , to_spacy_desc

nb_classes = len(labels)

#load the model to be tested.
model = load_model('backup/intent_models/my_model.h5')
def predict(sentence):
    
    n_test = 1
    test_vec_seq = [] 
    test_seq = [] 
    for i in range(n_test):
        test_text = sentence
        test_seq.append(test_text)
        test_doc = nlp(test_text)
        test_vec = []
        for word in test_doc:
            test_vec.append(word.vector)
        test_vec_seq.append(test_vec)
    test_vec_seq = pad_vec_sequences(test_vec_seq)
    prediction = model.predict(test_vec_seq)
    label_predictions = np.zeros(prediction.shape)
    
    for i in range(n_test):
        m = max(prediction[i])
        print (m)
        p = np.where(prediction[i] > 0.55 * m)	
        q = np.where(prediction[i] == m) and np.where(prediction[i] > 0.34)
        label_predictions[i][p] = 1
        label_predictions[i][q] = 2
        #print (p,q)
    for i in range(n_test):
        for x in range(len(label_predictions[i])):
            if label_predictions[i][x] == 2 :
                return sentence,labels[x]
            
        if  len(set(label_predictions[i])) == 1:
            return sentence,"NO INTENT"
            
    return sentence,"NO INTENT"

