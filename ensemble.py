import glob
import numpy as np
from utils import *

# Returns majority vote of the given .csv prediction files

NUM_PREDICTION_ROWS = 10000


def create_final_sub():

	pred_files = glob.glob('ensemble/*.csv')
	predictions = np.zeros((NUM_PREDICTION_ROWS, 2))
	for file in pred_files:
	    with open(file, 'r') as f:
	        lines = f.readlines()[1:]
	        current_preds = np.array([int(l.split(',')[1]) for l in lines])
	        current_preds[current_preds < 0] = 0
	        predictions[range(NUM_PREDICTION_ROWS), current_preds] += 1
	predictions = np.argmax(predictions, axis=1)
	predictions[predictions < 1 ] = -1
	create_csv_submission(predictions, 'output_ensemble_final.csv')


