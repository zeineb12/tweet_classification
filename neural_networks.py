import pandas as pd
import numpy as np
from numpy import asarray
from numpy import zeros
import re
import nltk
from nltk.corpus import stopwords
from numpy import array

#Keras import
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense, Masking
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D,MaxPooling1D,GRU
from keras.layers import Conv1D
from keras.layers import LSTM,Bidirectional
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer

batch_size = 1024
dim = 200
epochs = 6
validation_split = 0.05
verbose = 1


def build_simple_nn(X_train,y_train,X_test,y_test,vocab_size,embedding_matrix,maxlen):
     """
    Create the model for a Simple Neural Net
    INPUT:
        X_train : list         -
        y_train : list         -
        X_test : list          -
        y_test : list          -
        vocab_size :           -
        embedding_matrix :     -
        maxlen :               -    
    OUTPUT:
        Returns the model trained 
    """
    #Create model
    model = Sequential()
    embedding_layer = Embedding(vocab_size, dim, weights=[embedding_matrix], input_length=maxlen , trainable=False) #trainable set to False bc we use the downloaded dict
    model.add(embedding_layer)
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    
    #Fit model
    model.fit(X_train, y_train, batch_size, epochs, verbose, validation_split)
    
    score = model.evaluate(X_test, y_test, verbose)
    print("Test Score:", score[0])
    print("Test Accuracy:", score[1])
    return model

def build_rnn_lstm(X_train,y_train,X_test,y_test,vocab_size,embedding_matrix,maxlen):
    """
    Create the model for a long-short term memory network
    INPUT:
        X_train : list
        y_train : list
        X_test : list
        y_test : list
        vocab_size :
        embedding_matrix :
        maxlen :       
    OUTPUT:
        Returns the model trained and the history of the training
    """
    model = Sequential()
    model.add(
        Embedding(vocab_size, dim, weights=[embedding_matrix], input_length=maxlen,
                  trainable=False,
                  mask_zero=True))
    model.add(Masking(mask_value=0.0)) #need masking layer to not train on padding
    model.add(LSTM(64, return_sequences=False, 
                   dropout=0.1, recurrent_dropout=0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # Compile the model
    model.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, batch_size, epochs, verbose, validation_split)

    score = model.evaluate(X_test, y_test, verbose)
    print("Test Score:", score[0])
    print("Test Accuracy:", score[1])
    return model

def build_rnn_bi_lstm(X_train,y_train,X_test,y_test,vocab_size,embedding_matrix,maxlen):
    """
    Create the model for a Long-Short Term Memory network
    INPUT:
        X_train : list
        y_train : list
        X_test : list
        y_test : list
        vocab_size :
        embedding_matrix :
        maxlen :       
    OUTPUT:
        Returns the model trained and the history of the training
    """
    model = Sequential()
    model.add(Embedding(vocab_size, dim, weights=[embedding_matrix], input_length=maxlen , trainable=False))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    
    model.fit(X_train, y_train, batch_size, epochs, verbose, validation_split)

    score = model.evaluate(X_test, y_test, verbose)
    print("Test Score:", score[0])
    print("Test Accuracy:", score[1])
    return model




def build_rnn_gru(X_train,y_train,X_test,y_test,vocab_size,embedding_matrix,maxlen):
    """
    Create the model for a Convolutional Neural Network with Gated Recurrent Unit
    INPUT:
        X_train : list
        y_train : list
        X_test : list
        y_test : list
        vocab_size :
        embedding_matrix :
        maxlen :       
    OUTPUT:
        Returns the model trained and the history of the training
    """
    model = Sequential()
    embedding_layer = Embedding(vocab_size, dim, weights=[embedding_matrix], input_length=maxlen , trainable=False)
    model.add(embedding_layer)
    model.add(GRU(128, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    print(model.summary())
    
    model.fit(X_train, y_train, batch_size, epochs, verbose, validation_split)

    score = model.evaluate(X_test, y_test, verbose)
    print("Test Score:", score[0])
    print("Test Accuracy:", score[1])
    return model

def build_cnn(X_train,y_train,X_test,y_test,vocab_size,embedding_matrix,maxlen):
    """
    Create the model for a Convolutional Neural Network
    INPUT:
        X_train : list
        y_train : list
        X_test : list
        y_test : list
        vocab_size :
        embedding_matrix :
        maxlen :       
    OUTPUT:
        Returns the model trained and the history of the training
    """
    filters = 600
    kernel_size = 3

    model = Sequential()
    model.add(Embedding(vocab_size, dim, weights=[embedding_matrix], input_length=maxlen))
    model.add(Dropout(0.25))
    model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))
    model.add(Conv1D(300, kernel_size, padding='valid', activation='relu', strides=1))
    model.add(Conv1D(150, kernel_size, padding='valid', activation='relu', strides=1))
    model.add(Conv1D(75, kernel_size, padding='valid', activation='relu', strides=1))
    model.add(Flatten())
    model.add(Dense(600))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, batch_size, epochs, verbose, validation_split)

    score = model.evaluate(X_test, y_test, verbose)
    print("Test Score:", score[0])
    print("Test Accuracy:", score[1])
    return model





