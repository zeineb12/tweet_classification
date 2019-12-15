import pandas as pd
import csv
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize
import gensim 

def create_and_save_embeddings_vect(input_path,output_path,size,window,min_count,epochs=10):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: input_path (csv file of the tweets we want to create the word embeddings from)
               output_path (where we will save the txt file containing the embedding vectors)
               size (size of the dense vector to represent each word)
               window (maximum distance between the target word and its neighboring word)
               min_count (minimum frequency count of words)
               epochs
    """

    df = pd.read_csv(input_path)
    df = df['tweet']
    
    #tokenize the tweets to get a list of the list of the words of each tweet
    l_tweets = list(df.apply(lambda x : word_tokenize(x))) 
    
    model = gensim.models.Word2Vec (l_tweets, size, window, min_count, workers=10)
    model.train(l_tweets,total_examples=len(l_tweets),epochs)
    word_vectors = model.wv #get only the word embedding vectors from the model
    word_vectors.save_word2vec_format(output_path, binary=False) #export the word embeddings
    
    
    