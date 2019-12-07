import csv
import numpy as np

def open_by_tweets(path,sep="<user>"):
    """
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