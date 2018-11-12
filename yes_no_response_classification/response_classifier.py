# use natural language toolkit
import re
import numpy as np 
from random import randint
import pandas as pd
import string
import chardet

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def data_processing():
    with open('input.csv', 'rb') as f:
        result = chardet.detect(f.read())  # or readline if the file is large
    
    dataset=pd.read_csv('input.csv', encoding=result['encoding'])
    x=dataset.iloc[:,0]
    Y=dataset.iloc[:,1]
    x=x.to_dict()
    
    X=[]
    for d in range(len(x)):
        b=x[d].lower()
        sentence= re.sub(r'\d+','', b)
        sentence= re.sub('['+string.punctuation+']', '', sentence)
        X.append(sentence)
    return X,Y

count_vect=CountVectorizer()
tfidf_transformer = TfidfTransformer()

X, Y = data_processing()


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 0)

X_train_counts=count_vect.fit_transform(X_train)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.toarray()

def get_model():

    from sklearn.svm import SVC
    model= SVC(kernel = 'linear', random_state = 0)
    from sklearn.linear_model import LogisticRegression
    #model= LogisticRegression(penalty = "l1", C = 1.25, class_weight = "balanced")
    '''
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=2, weights = 'distance')
    '''
    
    '''
    from sklearn.naive_bayes import MultinomialNB
    model= MultinomialNB()
    '''
    '''
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors = 2, metric = 'minkowski', p = 2)
    '''
    '''
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    tuned_parameters = [{"C": [1,2,5,10,20,100], "kernel": ["linear"]}]
    clf = GridSearchCV(SVC(probability=True, class_weight='balanced',decision_function_shape='ovr'),
                                    param_grid=tuned_parameters,
                                    cv=5, scoring='f1_weighted', verbose=1)
    '''
    '''
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0)
    '''
    model.fit(X_train_tfidf, y_train)
    
    return model

model = get_model()

#model.score(X_train_tfidf, y_train)

def get_prediction(model):


    X_test_tfidf=count_vect.transform(X_test)

    y_pred=model.predict(X_test_tfidf)
    return y_pred


y_pred = get_prediction(model)
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score

def get_matrices(y_pred,y_test):
    
    cm = confusion_matrix(y_test, y_pred)
    Accuracy_Score = accuracy_score(y_test, y_pred)
    Recall=recall_score(y_test, y_pred, average='weighted')
    Precision=precision_score(y_test, y_pred, average='weighted')
    return cm,Accuracy_Score,Recall,Precision

cm,Accuracy_Score,Recall,Precision = get_matrices(y_pred,y_test)

#test1=pd.read_csv('data/no.txt', encoding=result['encoding']) 

def predict_response(response_query,model):
    cs1 = count_vect.transform([response_query])
    val = model.predict(cs1)
    predict_val = val.tolist()
    return predict_val


with open('data/yes.txt', 'rb') as f:
    test1 = f.readlines()

out = []

for i in test1:
    out.append(predict_response(i,model))
    
