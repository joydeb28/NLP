from preprocessor import Dataset, pad_vec_sequences, labels, pad_class_sequence
from sklearn import model_selection
import numpy as np

#from keras.preprocessing import sequence
from keras.layers import Dense, Dropout, Input, merge
from keras.layers.recurrent import LSTM
from keras import optimizers
from keras.layers import Embedding, Bidirectional,GRU
from keras.models  import Sequential, Model

def intialization():
    maxlen = 50 #sentences with length > maxlen will be ignored
    hidden_dim = 32
    nb_classes = len(labels)
    #initialise batch_size
    batch_size = 20
    #initialise num_epoch
    num_epoch = 25
    return maxlen,hidden_dim,nb_classes,batch_size,num_epoch

def make_data_set():
    ds = Dataset()
    print("Datasets loaded.")
    X_all = pad_vec_sequences(ds.X_all_vec_seq)
    Y_all = ds.Y_all
    #print (X_all.shape)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X_all,Y_all,test_size=0.2)
    y_train = pad_class_sequence(y_train, nb_classes)
    y_test = pad_class_sequence(y_test, nb_classes)
    y_test = np.array(y_test)
    x_train = np.asarray(x_train)
    x_train.ravel()
    
    y_train = np.asarray(y_train)
    y_train.ravel()
    return x_train,y_train,x_test,y_test


def create_model(maxlen,hidden_dim,nb_classes): 
    sequence = Input(shape=(maxlen,384), dtype='float32', name='input')
    forwards = LSTM(hidden_dim, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, dropout=0.1, recurrent_dropout=0.1)(sequence)
    backwards = LSTM(hidden_dim, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, dropout=0.1, recurrent_dropout=0.1, go_backwards=True)(sequence)
    merged = merge([forwards, backwards])
    
    after_dp = Dropout(0.6)(merged)
    output = Dense(nb_classes, activation='sigmoid', name='activation')(after_dp)

    model = Model(inputs=sequence, outputs=output)
    #Your model is now a bidirectional LSTM cell + a dropout layer + an activation layer.
    optimizers.Adam(lr=0.001, beta_1=0.6, beta_2=0.099, epsilon=1e-08, decay=0.005, clipnorm = 1., clipvalue = 0.5)
    model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy',metrics=['categorical_accuracy'])
    return model

def create_model_new(maxlen,hidden_dim,nb_classes):
    
    
    model=Sequential()
    model.add(Dropout(0.6, input_shape=(50, 384)))
    #model.add(Embedding(len(vocabulary), embedding_vecor_length, input_length=max_review_length))
    
    '''LSTM Model'''
    #model.add(LSTM(200,activation='relu'))
    
    '''Bidirectional LSTM Model'''
    model.add(Bidirectional((LSTM(200,activation='relu'))))
    
    '''GRU Model'''
    #model.add(GRU(200,activation='relu'))
    
    model.add(Dropout(0.015))
    model.add(Dense(20))
    # model.add(LSTM(200,activation='relu',return_sequences=True))
    # model.add(Dropout(0.005))
    # model.add(LSTM(100,activation='relu',return_sequences=True))
    # model.add(LSTM(50))
    
    model.add(Dense(nb_classes, activation='softmax'))
    #rmsprop=optimizers.rmsprop(lr=0.01)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print("Model compilation complete.")
    return model


def model_train(model,x_train,y_train,x_test,y_test,batch_size,num_epoch):

    print("Fitting to model")
    model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epoch, validation_data=[x_test, y_test])
    print("Model Training complete.")
    return model

def save_model(model):    
    model.save("backup/intent_models/model.h5")
    print("Model saved to Model folder.")
		
		
maxlen,hidden_dim,nb_classes,batch_size,num_epoch = intialization()
x_train,y_train,x_test,y_test = make_data_set()
model = create_model(maxlen,hidden_dim,nb_classes)
model = model_train(model,x_train,y_train,x_test,y_test,batch_size,num_epoch)
save_model(model)
