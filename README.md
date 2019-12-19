# ML_Project2 : Twitter Sentiment Analysis

## Team members

Sarah Antille: zeineb.sahnoun@epfl.ch

Lilia Ellouz: lilia.ellouz@epfl.ch   

Zeineb Sahnoun  zeineb.sahnoun@epfl.ch

## Introduction

This project is part of the Machine Learning course at EPFL. It is part of a challenge hosted on AIcrowd ( [https://www.aicrowd.com/challenges/epfl-ml-text-classification-2019](https://www.aicrowd.com/challenges/epfl-ml-text-classification-2019 )  ) . 

Given a training set of tweets labeled as expressing a happy feeling versus a sad feeling, we use NLP techniques as well as machine learning models to predict the sentiments of an unlabeled test dataset.

## Dataset Information
To obtain the same results than us, you need to download the following files from [https://www.aicrowd.com/challenges/epfl-ml-text-classification-2019/dataset_files](https://www.aicrowd.com/challenges/epfl-ml-text-classification-2019/dataset_files)  :

- `train_pos_full.txt` (contains the positive tweets with the happy smiley removed)
- `train_neg_full.txt` (contains the negative tweets with the sad smiley removed)
- `test_data.txt` (contains the tweets to predict)

**Make sure to have the above files in a directory called data to be able to run the scripts**

## Library requirements
In order to run the project, you need the following librairies installed:

- `scikit-learn`
- `keras` with backend `tensorflow` installed and configured
- `nltk`
- `gensim`
- `globe`
- `sklearn`
- `re`
- `pandas`
- `numpy`

## Files
- `run.py` : creates the .csv file used in our best prediction on AIcrowd

- `preprocessing.py`: contains the required methods to clean the training set and the test set

- `neural_networks.py` : contains the following neural nets algo:
	- simple neural net
	- recurrent neural net with long-short term memory 
	- recurrent neural net with bidirectional long-short term memory 
	- recurrent neural net with gated recurrent unit
	- convolutional neural network
    
- `ml_models.py` : trains and validates our classifiers and prints their accuracy on the validation set.
You should run this file as follows: ```$ python ml_models.py model_name``` where `model_name` can be one of the following:
	- **baseline**: for a [Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html) classifier that uses Count Vectorization
	- **bayes**: for a [Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html) classifier that uses TF-IDF vectorization
	- **sgd**: for a [Stochastic Gradient Descent Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)
	- **svm**: for a [Support Vector Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
	- **logistic**: for a regularized [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) classifier which is our **best performing model among this group**.
    
- `create_embeddings.py` : creates word2vec vectors from the dataset

- `ensemble.py` : computes the majority voting of the predictions of 3 different models

- `utils.py`

- `Rapport_ML_2.pdf` : 4 pages report explaining our approach and trials an errors


## Result:
- 0.880 accuracy on AIcrowd where we are ranked 8th among all the groups.



## Reproducibility

To obtain the same predictions we used for the AIcrowd submission, run the python script `run.py` . It will produce a file `submissions.csv` that can be submitted on the web page of the challenge.


## Remarks
- It will take a long time to run (few hours) because it pre-processes the dataset, then creates word2vec embedding vectors, and then run different neural network models.
- To speed up the neural network training, we used colab : [https://colab.research.google.com/notebooks/welcome.ipynb](https://colab.research.google.com/notebooks/welcome.ipynb)
