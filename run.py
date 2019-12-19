from utils import *
from preprocessing import *
from neural_networks import *
from embeddings import *

MAXLEN = 40
NUM_WORD = 120000

#load and clean the dataset
train_set, unknown = train_test_cleaner()
#create the word2vec model
create_and_save_embedding_vect(train_set)


#split training set
X_train, X_test, y_train, y_test = train_test_split(train_set['tweet'], train_set['label'], test_size=0.05, random_state=42)

#prepare input for neural nets
X_train,X_test, vocab_size, tokenizer = tok_and_pad_for_nn(X_train, X_test, MAXLEN, NUM_WORD)

#create embedding matrix from our word2vec model
embedding_matrix = create_embedding_matrix(vocab_size, tokenizer)


#Train and run first model + prediction on unknown set
model_1 = simple_nn(X_train, y_train,X_test,y_test,vocab_size,embedding_matrix, MAXLEN)
preds_1 = compute_predictions_nn(to_predict = unknown, threshold = 0.5, model = model_1, tokenizer = tokenizer,maxlen=MAXLEN)

#Train and run 2nd model + prediction on unknown set
model_2 = cnn(X_train, y_train,X_test,y_test,vocab_size,embedding_matrix, MAXLEN,first_dropout = 0.25)
preds_2 = compute_predictions_nn(to_predict = unknown, threshold = 0.5, model = model_2, tokenizer = tokenizer,maxlen=MAXLEN)

#Train and run 3rd model + prediction on unknown set
model_3 = rnn_gru(X_train, y_train,X_test,y_test,vocab_size,embedding_matrix, MAXLEN)
preds_3 = compute_predictions_nn(to_predict = unknown, threshold = 0.5, model = model_3, tokenizer = tokenizer,maxlen=MAXLEN)

#majority voting over the predictions of the 3 model









