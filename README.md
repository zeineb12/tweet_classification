# ML_Project2 : Twitter Sentiment Analysis

This project was part of the course Machine Learning at EPFL. It was part of a challenge hosted on AIcrowd ( [https://www.aicrowd.com/challenges/epfl-ml-text-classification-2019](https://www.aicrowd.com/challenges/epfl-ml-text-classification-2019 )  ) . Given two datasets (one with negative tweets and the other with positive tweets) we had to apply machine learning techniques to predict the sentiments of test dataset.

## Dataset Information
To obtain the same results than us, you need to download the following files from [https://www.aicrowd.com/challenges/epfl-ml-text-classification-2019/dataset_files](https://www.aicrowd.com/challenges/epfl-ml-text-classification-2019/dataset_files)  :

- `train_pos_full.txt` (contains the positive tweets with the happy smiley removed)
- `train_neg_full.txt` (contains the negative tweets with the sad smiley removed)
- `test_data.txt` (contains the tweets to predict)

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

- `run.py`
- `neural_networks.py` : contains the following neural nets algo:
	- simple neural net
	- recurrent neural net with long-short term memory 
	- recurrent neural net with bidirectional long-short term memory 
	- recurrent neural net with gated recurrent unit
	- convolutional neural network
- `create_embeddings.py` : creates word2vec vectors from the dataset
- `ensemble.py` : computes the majority voting of the predictions of 3 different models
- `utils.py`
- `Rapport_ML_2.pdf` : 4 pages report explaining 


## Team members

Sarah Antille
Lilia Ellouz
Zeineb Sahnoun

## Result:
- ?? % accuracy of prediction



## Run

To obtain the same predictions than us, run the python script `run.py` . It will produce a file `submissions.csv` that can be submitted on the web page of the challenge


## Remarks
- It will take a long time to run (few hours) because it pre-processes the dataset, then creates word2vec embedding vectors, and then run different neural network models.
- To speed up the neural network training, we used colab : [https://colab.research.google.com/notebooks/welcome.ipynb](https://colab.research.google.com/notebooks/welcome.ipynb)
