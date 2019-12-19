import sys

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

from preprocessing import *
SEED = 15432

def baseline(x_train,y_train,x_validation,y_validation):
	"""Trains a naive bayes model on count vectorized data and prints accuracy"""
	count_vect = CountVectorizer(max_features=80000,ngram_range=(1, 3))
	x_train_counts = count_vect.fit_transform(x_train)
	x_validation_counts = count_vect.transform(x_validation)
	clf = MultinomialNB().fit(x_train_counts, y_train)
	y_predicted = clf.predict(x_validation_counts)
	print(f'Baseline accuracy on validation set is {metrics.accuracy_score(y_validation, y_predicted)}')

def prepare_tfidf(x_train,y_train,x_validation,y_validation):
	tfidf_transformer = TfidfVectorizer(max_features=80000,ngram_range=(1, 3))
	x_train_tfidf = tfidf_transformer.fit_transform(x_train)
	x_validation_tfidf = tfidf_transformer.transform(x_validation)
	return x_train_tfidf,x_validation_tfidf

def bayes_model(x_train,y_train,x_validation,y_validation):
	"""Trains a naive bayes model on tfidf data and prints accuracy"""
	x_train_tfidf,x_validation_tfidf = prepare_tfidf(x_train,y_train,x_validation,y_validation)
	clf = MultinomialNB().fit(x_train_tfidf, y_train)
	y_predicted = clf.predict(x_validation_tfidf)
	print(f'NaiveBayes accuracy on validation set is {metrics.accuracy_score(y_validation, y_predicted)}')

def sgd_model(x_train,y_train,x_validation,y_validation):
	"""Trains a Stochastic Gradient Descent Classifier on tfidf data and prints accuracy"""
	x_train_tfidf,x_validation_tfidf = prepare_tfidf(x_train,y_train,x_validation,y_validation)
	clf = SGDClassifier(tol=1e-3, loss='modified_huber').fit(x_train_tfidf, y_train)
	y_predicted = clf.predict(x_validation_tfidf)
	print(f'SGD classifier accuracy on validation set is {metrics.accuracy_score(y_validation, y_predicted)}')

def svc_model(x_train,y_train,x_validation,y_validation):
	"""Trains an Support Vector Machine Classifier on tfidf data and prints accuracy"""
	x_train_tfidf,x_validation_tfidf = prepare_tfidf(x_train,y_train,x_validation,y_validation)
	clf = CalibratedClassifierCV(base_estimator=LinearSVC(penalty='l2', dual=False), cv=3).fit(x_train_tfidf, y_train)
	y_predicted = clf.predict(x_validation_tfidf)
	print(f'SVC accuracy on validation set is {metrics.accuracy_score(y_validation, y_predicted)}')

def logistic_model(x_train,y_train,x_validation,y_validation):
	"""Trains an Support Vector Machine Classifier on tfidf data and prints accuracy"""
	x_train_tfidf,x_validation_tfidf = prepare_tfidf(x_train,y_train,x_validation,y_validation)
	clf = LogisticRegression().fit(x_train_tfidf, y_train)
	y_predicted = clf.predict(x_validation_tfidf)
	print(f'Logistic Regression accuracy on validation set is {metrics.accuracy_score(y_validation, y_predicted)}')


if __name__ == "__main__":
	
	#get the model's name
	model = sys.argv[1]
	
	#Clean training set and test set
	train_set, test_set = train_test_cleaner()

	#Split into training set and validation set
	x = train_set.tweet
	y = train_set.label
	x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=.3, random_state=SEED)
	
	#Train the model selected and print accuracy on validation set
	if (model == 'baseline'):
		baseline(x_train,y_train,x_validation,y_validation)
	elif (model == 'bayes'):
		bayes_model(x_train,y_train,x_validation,y_validation)
	elif (model == 'sgd'):
		sgd_model(x_train,y_train,x_validation,y_validation)
	elif (model == 'svm'):
		svc_model(x_train,y_train,x_validation,y_validation)
	elif (model == 'logistic'):
		logistic_model(x_train,y_train,x_validation,y_validation)
	else:
		print('Invalid model name')


