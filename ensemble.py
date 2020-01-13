# Standard imports
import argparse
from pathlib import Path
import datetime as dt
import re
import pandas as pd
import os


# Preprocessing
from sklearn.base import BaseEstimator, TransformerMixin




parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", required=True, type=Path,
					help="project root directory")


def write_submission_file(predictions,model):

	submissions_dir = ROOT_DIR /"submissions"
	predicted_df = pd.DataFrame({"radiant_win_prob": predictions.values},index=predictions.index)
	submission_datetime = dt.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
	predicted_df.to_csv(submissions_dir/"submission_{}_{}.csv".format(submission_datetime,model))

if 	__name__ == "__main__":

	elapsed_time = dt.datetime.now()
	args = parser.parse_args()
	assert args.root_dir.is_dir()

	ROOT_DIR = args.root_dir
	SEED = 0

	df_test = pd.read_pickle(ROOT_DIR/"data"/"processed"/"X_test.pkl")

	models = ["LR","XGB","LGBM"]
	ensemble_preds = pd.DataFrame(index=df_test.index)
	for file_name in os.listdir(ROOT_DIR/"submissions"):
		file_path = ROOT_DIR/"submissions"/file_name
		model = re.search('[A-Z]+',file_name).group(0)
		preds = pd.read_csv(file_path)
		ensemble_preds[model] = preds["radiant_win_prob"].values

	ensemble_preds = ensemble_preds[models].mean(axis=1)
	write_submission_file(ensemble_preds,'ENSEMBLE')

	elapsed_time = dt.datetime.now() - elapsed_time
	print("Running took {} s".format(elapsed_time))