import numpy as np 
from validation import compute_f1
from keras.models import Model
from keras.layers import TimeDistributed,Conv1D,Dense,Embedding,Input,Dropout,LSTM,Bidirectional,MaxPooling1D,Flatten,concatenate
from prepro import readfile,createBatches,createMatrices,iterate_minibatches,addCharInformatioin,padding
from keras.utils import Progbar
from keras.initializers import RandomUniform

epochs = 300
fEmbeddings = open("embeddings/glove.6B.100d.txt", encoding="utf-8")

def tag_dataset(dataset,model):
   
    correctLabels = []
    predLabels = []
    b = Progbar(len(dataset))
    for i,data in enumerate(dataset):    
        tokens, casing,char, labels = data
        tokens = np.asarray([tokens])     
        casing = np.asarray([casing])
        char = np.asarray([char])
        pred = model.predict([tokens, casing,char], verbose=False)[0] 
        pred = pred.argmax(axis=-1) #Predict the classes            
        correctLabels.append(labels)
        predLabels.append(pred)
        b.update(i)
    return predLabels, correctLabels

def predict(sentence,model):
    sen_list = [[[i,'O\n'] for i in sentence.split()]]
    #sen_list = [[['SOCCER', 'O\n'], ['-', 'O\n'], ['JAPAN', 'O\n'], ['GET', 'O\n'], ['LUCKY', 'O\n'], ['WIN', 'O\n'], [',', 'O\n'], ['CHINA', 'O\n'], ['IN', 'O\n'], ['SURPRISE', 'O\n'], ['DEFEAT', 'O\n'], ['.', 'O\n']]]
    test = addCharInformatioin(sen_list)
    
    predLabels = []
    
    test_set = padding(createMatrices(test, word2Idx, label2Idx, case2Idx,char2Idx))
    
    test_batch,test_batch_len = createBatches(test_set)
    
    for i,data in enumerate(test_batch):    
        tokens, casing,char, labels = data

        tokens = np.asarray([tokens])     
        casing = np.asarray([casing])
        char = np.asarray([char])
        pred = model.predict([tokens, casing,char], verbose=False)[0] 
        pred = pred.argmax(axis=-1) #Predict the classes            
        predLabels.append(pred)
    entity_labels = []
    j = 0
    words_list = sentence.split()
    for i in predLabels[-1]:
        entity_labels.append((words_list[j],idx2Label[int(i)]))
        j+=1
    print("predLabels",entity_labels)    
    
    return entity_labels

def make_dataset(file_name):
    Senetnecs = readfile(file_name)
    Senetnecs = addCharInformatioin(Senetnecs)
    return Senetnecs


trainSentences = make_dataset("flight_data/train.txt")
devSentences = make_dataset("flight_data/valid.txt")
testSentences = make_dataset("flight_data/test.txt")

def create_index():

    labelSet = set()
    words = {}
    
    for dataset in [trainSentences, devSentences, testSentences]:
        for sentence in dataset:
            for token,char,label in sentence:
                labelSet.add(label)
                words[token.lower()] = True
    
    # :: Create a mapping for the labels ::
    label2Idx = {}
    for label in labelSet:
        label2Idx[label] = len(label2Idx)
    
    # :: Hard coded case lookup ::
    case2Idx = {'numeric': 0, 'allLower':1, 'allUpper':2, 'initialUpper':3, 'other':4, 'mainly_numeric':5, 'contains_digit': 6, 'PADDING_TOKEN':7}
    
    
    # :: Read in word embeddings ::
    word2Idx = {}
    wordEmbeddings = []
    
    
    for line in fEmbeddings:
        split = line.strip().split(" ")
        
        if len(word2Idx) == 0: #Add padding+unknown
            word2Idx["PADDING_TOKEN"] = len(word2Idx)
            vector = np.zeros(len(split)-1) #Zero vector vor 'PADDING' word
            wordEmbeddings.append(vector)
            
            word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
            vector = np.random.uniform(-0.25, 0.25, len(split)-1)
            wordEmbeddings.append(vector)
    
        if split[0].lower() in words:
            vector = np.array([float(num) for num in split[1:]])
            wordEmbeddings.append(vector)
            word2Idx[split[0]] = len(word2Idx)
            
    wordEmbeddings = np.array(wordEmbeddings)
    
    char2Idx = {"PADDING":0, "UNKNOWN":1}
    for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
        char2Idx[c] = len(char2Idx)
    
    return word2Idx,  label2Idx, case2Idx,char2Idx,wordEmbeddings

    
    
word2Idx,  label2Idx, case2Idx, char2Idx, wordEmbeddings = create_index()

train_set = padding(createMatrices(trainSentences,word2Idx,  label2Idx, case2Idx,char2Idx))
dev_set = padding(createMatrices(devSentences,word2Idx, label2Idx, case2Idx,char2Idx))
test_set = padding(createMatrices(testSentences, word2Idx, label2Idx, case2Idx,char2Idx))

idx2Label = {v: k for k, v in label2Idx.items()}

train_batch,train_batch_len = createBatches(train_set)
dev_batch,dev_batch_len = createBatches(dev_set)
test_batch,test_batch_len = createBatches(test_set)

caseEmbeddings = np.identity(len(case2Idx), dtype='float32')

def entity_model_BRNN():
    words_input = Input(shape=(None,),dtype='int32',name='words_input')
    words = Embedding(input_dim=wordEmbeddings.shape[0], output_dim=wordEmbeddings.shape[1],  weights=[wordEmbeddings], trainable=False)(words_input)
    casing_input = Input(shape=(None,), dtype='int32', name='casing_input')
    casing = Embedding(output_dim=caseEmbeddings.shape[1], input_dim=caseEmbeddings.shape[0], weights=[caseEmbeddings], trainable=False)(casing_input)
    character_input=Input(shape=(None,52,),name='char_input')
    embed_char_out=TimeDistributed(Embedding(len(char2Idx),30,embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)), name='char_embedding')(character_input)
    dropout= Dropout(0.5)(embed_char_out)
    conv1d_out= TimeDistributed(Conv1D(kernel_size=3, filters=30, padding='same',activation='tanh', strides=1))(dropout)
    maxpool_out=TimeDistributed(MaxPooling1D(52))(conv1d_out)
    char = TimeDistributed(Flatten())(maxpool_out)
    char = Dropout(0.5)(char)
    output = concatenate([words, casing,char])
    output = Bidirectional(LSTM(200, return_sequences=True, dropout=0.50, recurrent_dropout=0.25))(output)
    output = TimeDistributed(Dense(len(label2Idx), activation='softmax'))(output)
    model = Model(inputs=[words_input, casing_input,character_input], outputs=[output])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam')
    return model

#model.summary()
# plot_model(model, to_file='model.png')
entity_extract_model_BLSTM = entity_model_BRNN()
def train_model(model):
    
    for epoch in range(epochs):    
        print("Epoch %d/%d"%(epoch,epochs))
        a = Progbar(len(train_batch_len))
        for i,batch in enumerate(iterate_minibatches(train_batch,train_batch_len)):
            labels, tokens, casing,char = batch       
            model.train_on_batch([tokens, casing,char], labels)
            a.update(i)
        print(' ')
    return model

entity_extract_model_BLSTM = train_model(entity_extract_model_BLSTM)

def save_model(model,model_name):    
    model.save("Models/"+model_name+".h5")
    print("Model saved to Model folder.")

#   Performance on dev dataset        
predLabels, correctLabels = tag_dataset(dev_batch,entity_extract_model_BLSTM)        
pre_dev, rec_dev, f1_dev = compute_f1(predLabels, correctLabels, idx2Label)
print("Dev-Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (pre_dev, rec_dev, f1_dev))
    
#   Performance on test dataset       
predLabels, correctLabels = tag_dataset(test_batch,entity_extract_model_BLSTM)        
pre_test, rec_test, f1_test= compute_f1(predLabels, correctLabels, idx2Label)
print("Test-Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (pre_test, rec_test, f1_test))

save_model(entity_extract_model_BLSTM,"BRNN_Entity_Model")
#sentence ="a flight from BLR to MAA on 2018/07/30"
#predict(sentence,entity_extract_model_BLSTM)