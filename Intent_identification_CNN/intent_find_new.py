#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from preprocessor import Dataset, pad_vec_sequences, labels, pad_class_sequence
from sklearn import model_selection
import numpy as np

#from keras.preprocessing import sequence
#from keras.models import Model
from keras.layers import Dense, Input, merge, Embedding, Bidirectional,GRU,Concatenate,Reshape
#from keras.layers import Dropout
from keras.layers.recurrent import LSTM
#from keras import optimizers
from keras.layers import concatenate, Activation
#from keras.layers.wrappers import Bidirectional
from keras.layers.convolutional import Conv1D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalMaxPooling1D
from keras.models import Model
from keras.models  import Sequential
from keras.layers.pooling import GlobalMaxPooling1D, MaxPooling1D, GlobalAveragePooling1D
#from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from keras.layers.core import Dropout
from keras.regularizers import l2

from keras import backend as K
nb_classes = len(labels)
params = {
        
        "kernel_sizes_cnn": "1 2 3",
        "filters_cnn": 256,
        "embedding_size": 384,
        "lear_metrics": "binary_accuracy fmeasure",
        "confident_threshold": 0.5,
        "model_from_saved": False,
        "optimizer": "Adam",
        "lear_rate": 0.1,
        "lear_rate_decay": 0.1,
        "loss": "binary_crossentropy",
        "module": "fasttext",
        "text_size": 50,
        "coef_reg_cnn": 1e-4,
          "coef_reg_den": 1e-4,
          "dropout_rate": 0.5,
          "epochs": 2,
          "dense_size": 100,
          "model_name": "cnn_model",
          "batch_size": 64,
          "val_every_n_epochs": 5,
             "verbose": True,
          "val_patience": 5,
          "show_examples": False}
    
def intialization():
    maxlen = 50 #sentences with length > maxlen will be ignored
    hidden_dim = 32
    nb_classes = len(labels)
    #initialise batch_size
    batch_size = 5
    #initialise num_epoch
    num_epoch = 100
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

def cnn_model(params):
    inp = Input(shape=(params['text_size'], params['embedding_size']))
    outputs = []
    for i in range(len(params['kernel_sizes_cnn'])):
        output_i = Conv1D(params['filters_cnn'], kernel_size=params['kernel_sizes_cnn'][i],
                          activation=None,
                          kernel_regularizer=l2(params['coef_reg_cnn']),
                          padding='same')(inp)
        output_i = BatchNormalization()(output_i)
        output_i = Activation('relu')(output_i)
        output_i = GlobalMaxPooling1D()(output_i)
        outputs.append(output_i)

    output = concatenate(outputs, axis=1)
    output = Dropout(rate=params['dropout_rate'])(output)
    output = Dense(params['dense_size'], activation=None,
                    kernel_regularizer=l2(params['coef_reg_den']))(output)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    output = Dropout(rate=params['dropout_rate'])(output)
    output = Dense(nb_classes, activation=None,
                   kernel_regularizer=l2(params['coef_reg_den']))(output)
    output = BatchNormalization()(output)
    act_output = Activation("softmax")(output)
    model = Model(inputs=inp, outputs=act_output)
    model.compile(optimizer="Adam",
                      loss="binary_crossentropy",
                      metrics=['accuracy'],
                      #loss_weights=loss_weights,
                      #sample_weight_mode=sample_weight_mode,
                      # weighted_metrics=weighted_metrics,
                      # target_tensors=target_tensors
                      )
    return model
def dcnn_model(params):
    
    if type(params['kernel_sizes_cnn']) is str:
            params['kernel_sizes_cnn'] = [int(x) for x in
                                            params['kernel_sizes_cnn'].split(' ')]
    
    inp = Input(shape=(params['text_size'], params['embedding_size']))

    output = inp

    for i in range(len(params['kernel_sizes_cnn'])):
        output = Conv1D(params['filters_cnn'], kernel_size=params['kernel_sizes_cnn'][i],
                        activation=None,
                        kernel_regularizer=l2(params['coef_reg_cnn']),
                        padding='same')(output)
        output = BatchNormalization()(output)
        output = Activation('relu')(output)
        output = MaxPooling1D()(output)

    output = GlobalMaxPooling1D()(output)
    output = Dropout(rate=params['dropout_rate'])(output)
    output = Dense(params['dense_size'], activation=None,
                   kernel_regularizer=l2(params['coef_reg_den']))(output)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    output = Dropout(rate=params['dropout_rate'])(output)
    output = Dense(nb_classes, activation=None,
                   kernel_regularizer=l2(params['coef_reg_den']))(output)
    output = BatchNormalization()(output)
    act_output = Activation(params.get("last_layer_activation", "sigmoid"))(output)
    model = Model(inputs=inp, outputs=act_output)
    model.compile(optimizer="Adam",
                      loss="binary_crossentropy",
                      metrics=['accuracy'],
                      #loss_weights=loss_weights,
                      #sample_weight_mode=sample_weight_mode,
                      # weighted_metrics=weighted_metrics,
                      # target_tensors=target_tensors
                      )
    return model

def cnn_model_max_and_aver_pool(params):

    inp = Input(shape=(params['text_size'], params['embedding_size']))
    outputs = []
    for i in range(len(params['kernel_sizes_cnn'])):
        output_i = Conv1D(params['filters_cnn'], kernel_size=params['kernel_sizes_cnn'][i],
                          activation=None,
                          kernel_regularizer=l2(params['coef_reg_cnn']),
                          padding='same')(inp)
        output_i = BatchNormalization()(output_i)
        output_i = Activation('relu')(output_i)
        output_i_0 = GlobalMaxPooling1D()(output_i)
        output_i_1 = GlobalAveragePooling1D()(output_i)
        output_i = Concatenate()([output_i_0, output_i_1])
        outputs.append(output_i)

    output = concatenate(outputs, axis=1)

    output = Dropout(rate=params['dropout_rate'])(output)
    output = Dense(params['dense_size'], activation=None,
                   kernel_regularizer=l2(params['coef_reg_den']))(output)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    output = Dropout(rate=params['dropout_rate'])(output)
    output = Dense(nb_classes, activation=None,
                       kernel_regularizer=l2(params['coef_reg_den']))(output)
    output = BatchNormalization()(output)
    act_output = Activation(params.get("last_layer_activation", "sigmoid"))(output)
    model = Model(inputs=inp, outputs=act_output)
    model.compile(optimizer="Adam",
                      loss="binary_crossentropy",
                      metrics=['accuracy'],
                      #loss_weights=loss_weights,
                      #sample_weight_mode=sample_weight_mode,
                      # weighted_metrics=weighted_metrics,
                      # target_tensors=target_tensors
                      )
    return model

def bilstm_model(params):

    inp = Input(shape=(params['text_size'], params['embedding_size']))

    output = Bidirectional(LSTM(params['units_lstm'], activation='tanh',
                                  return_sequences=True,
                                  kernel_regularizer=l2(params['coef_reg_lstm']),
                                  dropout=params['dropout_rate'],
                                  recurrent_dropout=params['rec_dropout_rate']))(inp)

    output = GlobalMaxPooling1D()(output)
    output = Dropout(rate=params['dropout_rate'])(output)
    output = Dense(params['dense_size'], activation=None,
                   kernel_regularizer=l2(params['coef_reg_den']))(output)
    output = Activation('relu')(output)
    output = Dropout(rate=params['dropout_rate'])(output)
    output = Dense(nb_classes, activation=None,
                   kernel_regularizer=l2(params['coef_reg_den']))(output)
    act_output = Activation(params.get("last_layer_activation", "sigmoid"))(output)
    model = Model(inputs=inp, outputs=act_output)
    model.compile(optimizer="Adam",
                      loss="binary_crossentropy",
                      metrics=['accuracy'],
                      #loss_weights=loss_weights,
                      #sample_weight_mode=sample_weight_mode,
                      # weighted_metrics=weighted_metrics,
                      # target_tensors=target_tensors
                      )
    return model

def bilstm_bilstm_model(params):

    inp = Input(shape=(params['text_size'], params['embedding_size']))

    output = Bidirectional(LSTM(params['units_lstm_1'], activation='tanh',
                                return_sequences=True,
                                kernel_regularizer=l2(params['coef_reg_lstm']),
                                dropout=params['dropout_rate'],
                                recurrent_dropout=params['rec_dropout_rate']))(inp)

    output = Dropout(rate=params['dropout_rate'])(output)

    output = Bidirectional(LSTM(params['units_lstm_2'], activation='tanh',
                                return_sequences=True,
                                kernel_regularizer=l2(params['coef_reg_lstm']),
                                dropout=params['dropout_rate'],
                                recurrent_dropout=params['rec_dropout_rate']))(output)

    output = GlobalMaxPooling1D()(output)
    output = Dropout(rate=params['dropout_rate'])(output)
    output = Dense(params['dense_size'], activation=None,
                    kernel_regularizer=l2(params['coef_reg_den']))(output)
    output = Activation('relu')(output)
    output = Dropout(rate=params['dropout_rate'])(output)
    output = Dense(nb_classes, activation=None,
                    kernel_regularizer=l2(params['coef_reg_den']))(output)
    act_output = Activation(params.get("last_layer_activation", "sigmoid"))(output)
    model = Model(inputs=inp, outputs=act_output)
    model.compile(optimizer="Adam",
                      loss="binary_crossentropy",
                      metrics=['accuracy'],
                      #loss_weights=loss_weights,
                      #sample_weight_mode=sample_weight_mode,
                      # weighted_metrics=weighted_metrics,
                      # target_tensors=target_tensors
                      )
    return model

def bilstm_cnn_model(params):
 

    inp = Input(shape=(params['text_size'], params['embedding_size']))

    output = Bidirectional(LSTM(params['units_lstm'], activation='tanh',
                                return_sequences=True,
                                kernel_regularizer=l2(params['coef_reg_lstm']),
                                dropout=params['dropout_rate'],
                                recurrent_dropout=params['rec_dropout_rate']))(inp)

    output = Reshape(target_shape=(params['text_size'], 2 * params['units_lstm']))(output)
    outputs = []
    for i in range(len(params['kernel_sizes_cnn'])):
        output_i = Conv1D(params['filters_cnn'],
                          kernel_size=params['kernel_sizes_cnn'][i],
                          activation=None,
                          kernel_regularizer=l2(params['coef_reg_cnn']),
                          padding='same')(output)
        output_i = BatchNormalization()(output_i)
        output_i = Activation('relu')(output_i)
        output_i = GlobalMaxPooling1D()(output_i)
        outputs.append(output_i)

    output = Concatenate(axis=1)(outputs)
    output = Dropout(rate=params['dropout_rate'])(output)
    output = Dense(params['dense_size'], activation=None,
                   kernel_regularizer=l2(params['coef_reg_den']))(output)
    output = Activation('relu')(output)
    output = Dropout(rate=params['dropout_rate'])(output)
    output = Dense(nb_classes, activation=None,
                   kernel_regularizer=l2(params['coef_reg_den']))(output)
    act_output = Activation(params.get("last_layer_activation", "sigmoid"))(output)
    model = Model(inputs=inp, outputs=act_output)
    model.compile(optimizer="Adam",
                      loss="binary_crossentropy",
                      metrics=['accuracy'],
                      #loss_weights=loss_weights,
                      #sample_weight_mode=sample_weight_mode,
                      # weighted_metrics=weighted_metrics,
                      # target_tensors=target_tensors
                      )
    return model

def cnn_bilstm_model(params):
    inp = Input(shape=(params['text_size'], params['embedding_size']))

    outputs = []
    for i in range(len(params['kernel_sizes_cnn'])):
        output_i = Conv1D(params['filters_cnn'], kernel_size=params['kernel_sizes_cnn'][i],
                          activation=None,
                          kernel_regularizer=l2(params['coef_reg_cnn']),
                          padding='same')(inp)
        output_i = BatchNormalization()(output_i)
        output_i = Activation('relu')(output_i)
        output_i = MaxPooling1D()(output_i)
        outputs.append(output_i)

    output = concatenate(outputs, axis=1)
    output = Dropout(rate=params['dropout_rate'])(output)

    output = Bidirectional(LSTM(params['units_lstm'], activation='tanh',
                                return_sequences=True,
                                kernel_regularizer=l2(params['coef_reg_lstm']),
                                dropout=params['dropout_rate'],
                                recurrent_dropout=params['rec_dropout_rate']))(output)

    output = GlobalMaxPooling1D()(output)
    output = Dropout(rate=params['dropout_rate'])(output)
    output = Dense(params['dense_size'], activation=None,
                   kernel_regularizer=l2(params['coef_reg_den']))(output)
    output = Activation('relu')(output)
    output = Dropout(rate=params['dropout_rate'])(output)
    output = Dense(nb_classes, activation=None,
                   kernel_regularizer=l2(params['coef_reg_den']))(output)
    act_output = Activation(params.get("last_layer_activation", "sigmoid"))(output)
    model = Model(inputs=inp, outputs=act_output)
    model.compile(optimizer="Adam",
                      loss="binary_crossentropy",
                      metrics=['accuracy'],
                      #loss_weights=loss_weights,
                      #sample_weight_mode=sample_weight_mode,
                      # weighted_metrics=weighted_metrics,
                      # target_tensors=target_tensors
                      )
    return model

def bilstm_self_add_attention_model(params):

    inp = Input(shape=(params['text_size'], params['embedding_size']))
    output = Bidirectional(LSTM(params['units_lstm'], activation='tanh',
                                return_sequences=True,
                                kernel_regularizer=l2(params['coef_reg_lstm']),
                                dropout=params['dropout_rate'],
                                recurrent_dropout=params['rec_dropout_rate']))(inp)

    output = MaxPooling1D(pool_size=2, strides=3)(output)

    output = additive_self_attention(output, n_hidden=params.get("self_att_hid"),
                                     n_output_features=params.get("self_att_out"))
    output = GlobalMaxPooling1D()(output)
    output = Dropout(rate=params['dropout_rate'])(output)
    output = Dense(params['dense_size'], activation=None,
                   kernel_regularizer=l2(params['coef_reg_den']))(output)
    output = Activation('relu')(output)
    output = Dropout(rate=params['dropout_rate'])(output)
    output = Dense(nb_classes, activation=None,
                   kernel_regularizer=l2(params['coef_reg_den']))(output)
    act_output = Activation(params.get("last_layer_activation", "sigmoid"))(output)
    model = Model(inputs=inp, outputs=act_output)
    model.compile(optimizer="Adam",
                      loss="binary_crossentropy",
                      metrics=['accuracy'],
                      #loss_weights=loss_weights,
                      #sample_weight_mode=sample_weight_mode,
                      # weighted_metrics=weighted_metrics,
                      # target_tensors=target_tensors
                      )
    return model

def bilstm_self_mult_attention_model(params):
    
    inp = Input(shape=(params['text_size'], params['embedding_size']))

    output = Bidirectional(LSTM(params['units_lstm'], activation='tanh',
                                return_sequences=True,
                                kernel_regularizer=l2(params['coef_reg_lstm']),
                                dropout=params['dropout_rate'],
                                recurrent_dropout=params['rec_dropout_rate']))(inp)

    output = MaxPooling1D(pool_size=2, strides=3)(output)

    output = multiplicative_self_attention(output, n_hidden=params.get("self_att_hid"),
                                           n_output_features=params.get("self_att_out"))
    output = GlobalMaxPooling1D()(output)
    output = Dropout(rate=params['dropout_rate'])(output)
    output = Dense(params['dense_size'], activation=None,
                   kernel_regularizer=l2(params['coef_reg_den']))(output)
    output = Activation('relu')(output)
    output = Dropout(rate=params['dropout_rate'])(output)
    output = Dense(nb_classes, activation=None,
                   kernel_regularizer=l2(params['coef_reg_den']))(output)
    act_output = Activation(params.get("last_layer_activation", "sigmoid"))(output)
    model = Model(inputs=inp, outputs=act_output)
    model.compile(optimizer="Adam",
                      loss="binary_crossentropy",
                      metrics=['accuracy'],
                      #loss_weights=loss_weights,
                      #sample_weight_mode=sample_weight_mode,
                      # weighted_metrics=weighted_metrics,
                      # target_tensors=target_tensors
                      )
    return model

def bigru_model(params):

    inp = Input(shape=(params['text_size'], params['embedding_size']))

    output = Bidirectional(GRU(params['units_lstm'], activation='tanh',
                               return_sequences=True,
                               kernel_regularizer=l2(params['coef_reg_lstm']),
                               dropout=params['dropout_rate'],
                               recurrent_dropout=params['rec_dropout_rate']))(inp)

    output = GlobalMaxPooling1D()(output)
    output = Dropout(rate=params['dropout_rate'])(output)
    output = Dense(params['dense_size'], activation=None,
                   kernel_regularizer=l2(params['coef_reg_den']))(output)
    output = Activation('relu')(output)
    output = Dropout(rate=params['dropout_rate'])(output)
    output = Dense(nb_classes, activation=None,
                   kernel_regularizer=l2(params['coef_reg_den']))(output)
    act_output = Activation(params.get("last_layer_activation", "sigmoid"))(output)
    model = Model(inputs=inp, outputs=act_output)
    model.compile(optimizer="Adam",
                      loss="binary_crossentropy",
                      metrics=['accuracy'],
                      #loss_weights=loss_weights,
                      #sample_weight_mode=sample_weight_mode,
                      # weighted_metrics=weighted_metrics,
                      # target_tensors=target_tensors
                      )
    return model
def lstm_model(params):
    
    model=Sequential()
    model.add(Dropout(0.6, input_shape=(params['text_size'], params['embedding_size'])))
    #model.add(Embedding(len(vocabulary), embedding_vecor_length, input_length=max_review_length))
    model.add(LSTM(32,activation='relu'))
    #model.add(Bidirectional((LSTM(200,activation='relu'))))
    model.add(Dropout(0.015))
    model.add(Dense(20))
    # model.add(LSTM(200,activation='relu',return_sequences=True))
    # model.add(Dropout(0.005))
    # model.add(LSTM(100,activation='relu',return_sequences=True))
    # model.add(LSTM(50))
    
    model.add(Dense(nb_classes, activation='softmax'))
    #rmsprop=optimizers.rmsprop(lr=0.01)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    #model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=[f1.precision,f1.recall,f1.f1Score])
    return model    
    
def model_train(model,x_train,y_train,x_test,y_test,batch_size,num_epoch):

    print("Fitting to model")
    model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epoch, validation_data=[x_test, y_test])
    print("Model Training complete.")
    return model

def save_model(model,model_name):    
    model.save("backup/intent_models/"+model_name+".h5")
    print("Model saved to Model folder.")
		
		
maxlen,hidden_dim,nb_classes,batch_size,num_epoch = intialization()
x_train,y_train,x_test,y_test = make_data_set()
#model = create_model(maxlen,hidden_dim,nb_classes)
model1 = cnn_model(params)
model1 = model_train(model1,x_train,y_train,x_test,y_test,batch_size,num_epoch)
save_model(model1,cnn_model)

model2 = dcnn_model(params)
model2 = model_train(model2,x_train,y_train,x_test,y_test,batch_size,num_epoch)
save_model(model2,dcnn_model)

model3 = cnn_model_max_and_aver_pool(params)
model3 = model_train(model3,x_train,y_train,x_test,y_test,batch_size,num_epoch)
save_model(model3,cnn_model_max_and_aver_pool)

model4 = bilstm_model(params)
model4 = model_train(model4,x_train,y_train,x_test,y_test,batch_size,num_epoch)
save_model(model4,bilstm_model)

model5 = bilstm_bilstm_model(params)
model5 = model_train(model5,x_train,y_train,x_test,y_test,batch_size,num_epoch)
save_model(model5,bilstm_bilstm_model)

model6 = bilstm_cnn_model(params)
model6 = model_train(model6,x_train,y_train,x_test,y_test,batch_size,num_epoch)
save_model(model6,bilstm_cnn_model)

model7 = cnn_bilstm_model(params)
model7 = model_train(model7,x_train,y_train,x_test,y_test,batch_size,num_epoch)
save_model(model7,cnn_bilstm_model)
'''
model8 = bilstm_self_add_attention_model(params)
model8 = model_train(model8,x_train,y_train,x_test,y_test,batch_size,num_epoch)
save_model(model8,bilstm_self_add_attention_model)

model9 = bilstm_self_mult_attention_model(params)
model9 = model_train(model9,x_train,y_train,x_test,y_test,batch_size,num_epoch)
save_model(model9,bilstm_self_mult_attention_model)
'''
model10 = bigru_model(params)
model11 = model_train(model10,x_train,y_train,x_test,y_test,batch_size,num_epoch)
save_model(model11,bigru_model)

model11 = lstm_model(params)
model11 = model_train(model11,x_train,y_train,x_test,y_test,batch_size,num_epoch)
save_model(model11,lstm_model)


