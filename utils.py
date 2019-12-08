import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

def open_by_tweets(path,sep="<user>"):
    """
    !!!!!DO NOT USE IT!!!!!
    Open txt file and output list of strings based on the separator, also remove '\n'
    """
    with open(path,"r") as file:
        train_pos = file.read().replace('\n','').split(sep)
    return train_pos

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w', newline='') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

def merge_datasets_with_labels(df_pos, df_neg):
    """
    Create a dataframe with columns "tweet" and "label" from the two given dataframes
    Arguments: df_pos (dataframe of the positive tweets)
               df_neg (dataframe of the negative tweets)
    """
    df_pos['label'] = 1 #label 1 for happy tweet
    df_neg['label'] = 0
    df_tweets = df_pos.append(df_neg) #our labelled dataset
    return df_tweets
            
def create_tfidf_clf(df_tweets, ngram = (2,3)):
    """
    Creates a linear classifer based on tfidf and returns the classifier and the count vector (it also prints the accuracy on the df)
    Arguments: df_tweets (dataframe for training with two columns, "tweet" and "label")
    """
    
    X_train, X_test, y_train, y_test = train_test_split(df_tweets['tweet'], df_tweets['label'], random_state = 0) 
    count_vect = CountVectorizer(ngram_range= ngram)
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = MultinomialNB().fit(X_train_tfidf, y_train)
    predicted = clf.predict(count_vect.transform(X_test))
    print(f'Our model\'s accuracy is {metrics.accuracy_score(y_test, predicted)}')
    
    return clf, count_vect
                
def test_dataset_to_df(path):
    """
    Opens the .txt file and returns a dataframe that keeps the correct id
    """
    df = pd.read_csv(path, sep="\n", header=None)
    df.columns = ["tweet"]
    df.index += 1 #index starting at 1 instead of 0 -> need that to keep the correct id of each tweet
    df['tweet'] = df['tweet'].apply(lambda x : str(x).split(',', maxsplit=1)[1]) #remove the first number in the each entry of the df
    #df= pd.DataFrame({'tweet' : df['tweet']}) #create the dataframe
    return df

def predict_tfidf_clf(df_to_predict, clf, count_vect, output_name):
    """
    Compute the predictions for the .txt file from path based on the given clf and count vector and outputs the csv file to submit on Kaggle
    """
    df_unknown = clf.predict(count_vect.transform(df_to_predict.values))
    df_unknown[df_unknown == 0] = -1 #replace 0 to -1
    create_csv_submission([x for x in range(1,len(df_unknown)+1)],df_unknown,output_name)
    