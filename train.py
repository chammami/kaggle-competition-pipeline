# Standard imports
import argparse
from pathlib import Path
import datetime as dt
import re
import pandas as pd
import numpy


# Preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, ShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# ML models
import xgboost
import lightgbm
import catboost
from sklearn.neighbors import NearestNeighbors
from sklearn.naive_bayes import BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import  RandomForestClassifier



parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", required=True, type=Path,
					help="project root directory")


def write_submission_file(predictions,model):

	submissions_dir = ROOT_DIR /"submissions"
	predicted_df = pd.DataFrame({"radiant_win_prob": predictions.values},index=predictions.index)
	submission_datetime = dt.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
	predicted_df.to_csv(submissions_dir/"submission_{}_{}.csv".format(submission_datetime,model))


class Identity(BaseEstimator, TransformerMixin):
	"""
	Add new attributes to training and test set.
	"""
	def fit(self, X, y=None):
		return self
	def transform(self, X, y=None):

		return X

def compute_cv_score(model):

	# rfc = RandomForestClassifier(n_estimators=400,max_depth=5)
	cv_scores = cross_val_score(model, X_train, y_train, cv=cv_scheme,scoring='roc_auc', n_jobs=-1)

	print(f"Cross validation AUC:{cv_scores}")
	print(f"Mean AUC:{cv_scores.mean()}, std {cv_scores.std()}")

if 	__name__ == "__main__":

	elapsed_time = dt.datetime.now()
	args = parser.parse_args()
	assert args.root_dir.is_dir()

	ROOT_DIR = args.root_dir
	SEED = 0

	df_train = pd.read_pickle(ROOT_DIR/"data"/"processed"/"X_train.pkl")
	df_test = pd.read_pickle(ROOT_DIR/"data"/"processed"/"X_test.pkl")
	y_train = pd.read_pickle(ROOT_DIR/"data"/"processed"/"y_test.pkl").values.astype(int)

	# Cross-validation scheme
	cv_scheme = ShuffleSplit(n_splits=10,random_state=SEED)

	pat = '^[0-9]+'
	unscaled_features = set([f for f in df_train.columns if re.match(pat,f)])
	scaled_features = list(set(df_train.columns) - unscaled_features)
	unscaled_features = list(unscaled_features)
	unscaled_features.extend([f for f in df_train.columns if "game_mode" in f])


	pipeline = Pipeline([("imputer",SimpleImputer()),
						 ('scaler',StandardScaler())])

	X_train = pipeline.fit_transform(df_train)
	X_test = pipeline.transform(df_test)


	# Modeling + Model selection + HO + Feature selection + Stacking + Ensembling
	logit = LogisticRegression(C=1, random_state=SEED, solver='liblinear')
	rf = RandomForestClassifier(n_estimators=100, max_depth=4,random_state=SEED)
	nbc = BernoulliNB()
	lda = LinearDiscriminantAnalysis()
	xgb = xgboost.XGBClassifier()
	lgbm = lightgbm.LGBMClassifier()
	cb = catboost.CatBoostClassifier()
	knn = NearestNeighbors(n_neighbors=100)

	models = {"LR":logit,
			  "RF":rf,
			  "NB": nbc,
			  # "LDA":lda,
			  # "XGB":xgb,
			  # "KNN":knn ,
			  # "CatBoost":cb,
			  "LGBM":lgbm
			  }
	for m in models:
		training_time = dt.datetime.now()
		print(m)
		model = models[m]
		compute_cv_score(model)
		# Train on the whole training data set
		model.fit(X_train, y_train)
		# Prediction
		preds = pd.Series(model.predict_proba(X_test)[:,1],index=df_test.index)

		# Submission
		write_submission_file(preds,m)
		training_time = dt.datetime.now() - training_time
		print("Training took {} s".format(training_time))

	elapsed_time = dt.datetime.now() - elapsed_time
	print("Running took {} s".format(elapsed_time))