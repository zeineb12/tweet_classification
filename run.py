from utils import *
from preprocessing import *
from neural_networks import *
from embeddings import *
from ensemble import *

MAXLEN_140 = 140
MAXLEN_40 = 40
NUM_WORD = 120000

#load and clean the dataset
train_set, unknown = train_test_cleaner()
#Shuffle the order of the entries of the train set (to mix the labels)
train_set.sample(frac=1, random_state=1).reset_index(drop=True)
#create the word2vec model
create_and_save_embedding_vect(train_set)


#split training set
X_train, X_test, y_train, y_test = train_test_split(train_set['tweet'], train_set['label'], test_size=0.05, random_state=42)

X_train, X_test, y_train, y_test = X_train, X_test, y_train, y_test
#prepare input for neural nets
X_train_140,X_test_140, vocab_size, tokenizer_140 = tok_and_pad_for_nn(X_train, X_test, MAXLEN_140, NUM_WORD)
X_train_40,X_test_40, vocab_size, tokenizer_40 = tok_and_pad_for_nn(X_train, X_test, MAXLEN_40, NUM_WORD)

#create embedding matrix from our word2vec model
embedding_matrix = create_embedding_matrix(vocab_size, tokenizer)


#Train and run first model + prediction on unknown set
#4-convolutional neural net
model_1 = cnn(X_train_40, y_train_40,X_test_40,y_test_40,vocab_size_40,embedding_matrix, MAXLEN_40,first_dropout = 0.40)
preds_1 = compute_predictions_nn(to_predict = unknown, threshold = 0.51, model = model_1, tokenizer = tokenizer_40,maxlen=MAXLEN_140)
create_csv_submission(preds_1, "./ensemble/model1_pred.csv")

#Train and run 2nd model + prediction on unknown set
#4-convolutional neural net
model_2 = cnn(X_train_140, y_train_140,X_test_140,y_test_140,vocab_size_140,embedding_matrix, MAXLEN_140,first_dropout = 0.25)
preds_2 = compute_predictions_nn(to_predict = unknown, threshold = 0.5, model = model_2, tokenizer = tokenizer_140,maxlen=MAXLEN_140)
create_csv_submission(preds_2, "./ensemble/model2_pred.csv")

#Train and run 3rd model + prediction on unknown set
#recurrent neural network with bidirection long-short term memory
model_3 = rnn_bilstm(X_train_140, y_train_140,X_test_140,y_test_140,vocab_size_140,embedding_matrix, MAXLEN_140)
preds_3 = compute_predictions_nn(to_predict = unknown, threshold = 0.5, model = model_3, tokenizer = tokenizer_140,maxlen=MAXLEN_140)
create_csv_submission(preds_3, "./ensemble/model3_pred.csv")

#majority voting over the predictions of the 3 model, outputs the cvs file 'output_ensemble_final.csv'
create_final_sub()









