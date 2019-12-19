import pandas as pd
import csv
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize
import gensim 
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from numpy import zeros
from numpy import asarray


def create_and_save_embedding_vect(df):
    """
    Creates and train a Word2Vec model from our corpus of words
    """

    #df = pd.read_csv("./data/tweets_full.csv")
    #df = df['tweet']
    
    #tokenize the tweets to get a list of the list of the words of each tweet
    l_tweets = list(df['tweet'].apply(lambda x : word_tokenize(x))) 
    
    
    #size = vector dimension for each word
    #window = size of window neighborhood 
    #min_count = need the word to appear at least that number to be taken in the model
    #workers = number of cores used to run the model
    
    model = gensim.models.Word2Vec (l_tweets, size=200, window=10, min_count=5, workers=10)
    model.train(l_tweets,total_examples=len(l_tweets),epochs=10)
    word_vectors = model.wv #get only the word embedding vectors from the model
    word_vectors.save_word2vec_format('embedding_vec.txt', binary=False) #export the word embeddings vectors


def create_embedding_matrix(vocab_size, tokenizer):
    embeddings_dictionary = dict()
    glove_file = open('embedding_vec.txt', encoding="utf8")
    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dim = asarray(records[1:], dtype='float32')
        embeddings_dictionary [word] = vector_dim
    glove_file.close()

    embedding_matrix = zeros((vocab_size, 200))
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
    return embedding_matrix
    

    
def tok_and_pad_for_nn(X_train, X_test,maxlen,num_words):
    """ 
    Tokenizes and pads to maxlen each tweet for each given dataframe
    so that it can directly be used in a neural network
    INPUT: 
        X_train: Panda Series       - tweets
        X_test: Panda Series        - tweets
        to_predict: Panda Series    - tweets we have to label to submit to aicrowd
        maxlen: int                 - maximal length of word we take per tweet
        num_words: int              - the maximum number of words to keep, based on word frequency
    
    OUTPUT:
        X_train: ndarray            
        X_test: ndarray             
        to_predict: ndarray         
        vocab_size: int             
    """
    tokenizer = Tokenizer(num_words)
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)#convert each word to a integer based on the tokenizer
    X_test = tokenizer.texts_to_sequences(X_test)
    #to_predict = tokenizer.texts_to_sequences(to_predict)
    # Adding 1 because of reserved 0 index
    vocab_size = len(tokenizer.word_index) + 1
    
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen) #makes sure all tweets have 100 words (padding)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
    #to_predict = pad_sequences(to_predict, padding='post', maxlen=maxlen)

    return X_train, X_test, vocab_size, tokenizer

def tok_and_pad_for_unknown(unknown):
    return 0
    