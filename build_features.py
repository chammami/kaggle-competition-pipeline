from sklearn.base import BaseEstimator, TransformerMixin
import argparse
from pathlib import Path
import pandas as pd
import datetime as dt
import re
from sklearn.feature_extraction.text import CountVectorizer
import collections
import os
from tqdm import tqdm
import json

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", required=True, type=Path,
					help="project root directory")

MATCH_FEATURES = [
	('game_time', lambda m: m['game_time']),
	('game_mode', lambda m: m['game_mode']),
	('lobby_type', lambda m: m['lobby_type']),
	('objectives_len', lambda m: len(m['objectives'])),
	('chat_len', lambda m: len(m['chat'])),
]

PLAYER_FIELDS = [
	'hero_id',

	'kills',
	'deaths',
	'assists',
	'denies',

	'gold',
	'lh',
	'xp',
	'health',
	'max_health',
	'max_mana',
	'level',

	'x',
	'y',

	'stuns',
	'creeps_stacked',
	'camps_stacked',
	'rune_pickups',
	'firstblood_claimed',
	'teamfight_participation',
	'towers_killed',
	'roshans_killed',
	'obs_placed',
	'sen_placed',
]


def read_matches(matches_file):

	MATCHES_COUNT = {
		'test_matches.jsonl': 10000,
		'train_matches.jsonl': 39675,
	}
	_, filename = os.path.split(matches_file)
	total_matches = MATCHES_COUNT.get(filename)

	with open(matches_file) as fin:
		for line in tqdm(fin, total=total_matches):
			yield json.loads(line)


def extract_features_csv(match):
	row = [('match_id_hash', match['match_id_hash']),]

	for field, f in MATCH_FEATURES:
		row.append((field, f(match)))

	for slot, player in enumerate(match['players']):
		if slot < 5:
			player_name = 'r%d' % (slot + 1)
		else:
			player_name = 'd%d' % (slot - 4)

		for field in PLAYER_FIELDS:
			column_name = '%s_%s' % (player_name, field)
			row.append((column_name, player[field]))
		row.append((f'{player_name}_ability_level', len(player['ability_upgrades'])))
		row.append((f'{player_name}_max_hero_hit', player['max_hero_hit']['value']))
		row.append((f'{player_name}_purchase_count', len(player['purchase_log'])))
		row.append((f'{player_name}_count_ability_use', sum(player['ability_uses'].values())))
		row.append((f'{player_name}_damage_dealt', sum(player['damage'].values())))
		row.append((f'{player_name}_damage_received', sum(player['damage_taken'].values())))
		row.append( (f'{player_name}_items', list(map(lambda x: x['id'][5:], player['hero_inventory'])) ) )


	return collections.OrderedDict(row)

def extract_targets_csv(match, targets):
	return collections.OrderedDict([('match_id_hash', match['match_id_hash'])] + [
		(field, targets[field])
		for field in ['game_time', 'radiant_win', 'duration', 'time_remaining', 'next_roshan_team']
	])


def load_data():
	data_path = ROOT_DIR/"data"/"raw"
	train_features = pd.read_csv(data_path/"train_features.csv", index_col='match_id_hash')
	train_labels = pd.read_csv(data_path/"train_targets.csv", index_col='match_id_hash')
	df_train = pd.merge(train_labels[["radiant_win"]],train_features,how="inner",left_index=True, right_index=True)
	df_test = pd.read_csv(data_path/"test_features.csv", index_col='match_id_hash')

	return df_train, df_test

def extract_features_from_raw_data(set_type):

	df_new_features = []
	df_new_targets = []

	for match in read_matches(ROOT_DIR/"data"/"raw"/"{}_matches.jsonl".format(set_type)):

		match_id_hash = match['match_id_hash']
		features = extract_features_csv(match)
		df_new_features.append(features)

		if set_type =="train":
			targets = extract_targets_csv(match, match['targets'])
			df_new_targets.append(targets)


	df_new_features = pd.DataFrame.from_records(df_new_features).set_index('match_id_hash')

	if set_type =="train":
		df_new_targets = pd.DataFrame.from_records(df_new_targets).set_index('match_id_hash')
		df = pd.merge(df_new_targets[["radiant_win"]],df_new_features,how="inner",left_index=True, right_index=True)
	elif set_type =="test":
		df = df_new_features.copy()
	return df


class AttributesAdder(BaseEstimator, TransformerMixin):
	"""
	Add new attributes to training and test set.
	"""
	def fit(self, X, y=None):
		return self
	def transform(self, X, y=None):

		print("Building player features")
		player_data = pd.DataFrame()
		pat = '^(d|r)[0-9]_'
		features = set([re.sub(pat,"",f) for f in X.columns if re.match(pat,f)])
		features = features - set(["x","y","hero_id"])

		# Aggregate features with simple sums
		for t in ("r","d"):
			for f in features:
				player_data[f"{t}_{f}_sum"] = X[[f"{t}{i}_{f}" for i in  range(1,6)]].sum(axis=1)

		for f in features:
			player_data[f"{f}_diff"] = player_data[f"r_{f}_sum"] - player_data[f"d_{f}_sum"]

		# Leaving only the diff variables
		pat = 'diff'
		used_cols = set([f for f in player_data.columns if pat in f])
		player_features = player_data[used_cols]
		# print(used_cols)

		print("Encoding hero_id features")
		hero_id_data = X.copy()
		r_features = [f"r{i}_hero_id" for i in range(1,6)]
		d_features = [f"d{i}_hero_id" for i in range(1,6)]
		hero_id_data["r_hero_ids"] = hero_id_data.apply(lambda row: ' '.join(row.loc[r_features].map(int).map(str)), axis=1)
		hero_id_data["d_hero_ids"] = hero_id_data.apply(lambda row: ' '.join(row.loc[d_features].map(int).map(str)), axis=1)
		r_cvv = CountVectorizer()
		r_heroes = pd.DataFrame(r_cvv.fit_transform(hero_id_data['r_hero_ids']).todense(),
								columns=r_cvv.get_feature_names(),
								index=hero_id_data.index)
		r_heroes.sort_index(axis=1, inplace=True)

		d_cvv = CountVectorizer()
		d_heroes = pd.DataFrame(d_cvv.fit_transform(hero_id_data['d_hero_ids']).todense(),
								columns=d_cvv.get_feature_names(),
								index=hero_id_data.index)
		d_heroes.sort_index(axis=1, inplace=True)
		heroes_features = pd.merge(r_heroes,d_heroes,how="inner",left_index=True,right_index=True,suffixes=('_r','_d'))


		print("Encoding game mode feature")
		game_mode_features = pd.get_dummies(X["game_mode"], prefix="game_mode")
		game_mode_features.sort_index(axis=1, inplace=True)


		print("Adding match features")
		m_features = ["lobby_type","game_time","objectives_len","chat_len"]

		match_features = X[m_features].copy()

		X = pd.merge(player_features,heroes_features,how="inner",left_index=True,right_index=True)
		X = pd.merge(X,match_features,how="inner",left_index=True,right_index=True)
		X = pd.merge(X,game_mode_features,how="inner",left_index=True,right_index=True)

		return X

def add_items_dummies(train_df, test_df):

	full_df = pd.concat([train_df, test_df], sort=False)
	train_size = train_df.shape[0]

	for team in 'r', 'd':
		players = [f'{team}{i}' for i in range(1, 6)]
		item_columns = [f'{player}_items' for player in players]

		d = pd.get_dummies(full_df[item_columns[0]].apply(pd.Series).stack()).sum(level=0, axis=0)
		dindexes = d.index.values
		print(dindexes[:10])

		for c in item_columns[1:]:
			d = d.add(pd.get_dummies(full_df[c].apply(pd.Series).stack()).sum(level=0, axis=0), fill_value=0)
			d = d.ix[dindexes]

		full_df = pd.concat([full_df, d.add_prefix(f'{team}_item_')], axis=1, sort=False)
		full_df.drop(columns=item_columns, inplace=True)

	train_df = full_df.iloc[:train_size, :]
	test_df = full_df.iloc[train_size:, :]

	return train_df, test_df


def drop_consumble_items(train_df, test_df):

	full_df = pd.concat([train_df, test_df], sort=False)
	train_size = train_df.shape[0]

	for team in 'r', 'd':
		consumble_columns = ['tango', 'tpscroll',
							 'bottle', 'flask',
							 'enchanted_mango', 'clarity',
							 'faerie_fire', 'ward_observer',
							 'ward_sentry']

		starts_with = f'{team}_item_'
		consumble_columns = [starts_with + column for column in consumble_columns]
		full_df.drop(columns=consumble_columns, inplace=True)

	train_df = full_df.iloc[:train_size, :]
	test_df = full_df.iloc[train_size:, :]

	return train_df, test_df



if 	__name__ == "__main__":

	elapsed_time = dt.datetime.now()
	args = parser.parse_args()
	assert args.root_dir.is_dir()

	ROOT_DIR = args.root_dir
	SEED = 0
	print("Extracting features for train set")
	df_train = extract_features_from_raw_data(set_type="train")
	print("Extracting features for test set")
	df_test = extract_features_from_raw_data(set_type="test")


	print("Persisting features as pickle files")
	df_train.to_pickle(ROOT_DIR/"data"/"processed"/"df_train_raw.pkl")
	df_test.to_pickle(ROOT_DIR/"data"/"processed"/"df_test_raw.pkl")


	print("Dropping game mode 16")
	df_train = df_train[df_train.game_mode != 16].copy()
	y_train = df_train.radiant_win
	df_train = df_train.drop(["radiant_win"],axis=1)

	print("Processing items features")
	r_items_features = [f"r{i}_items" for i in range(1,6)]
	d_items_features = [f"d{i}_items" for i in range(1,6)]
	items_features = r_items_features + d_items_features
	df_train_items = df_train[items_features].copy()
	df_test_items = df_test[items_features].copy()
	df_train.drop(items_features, axis=1, inplace=True)
	df_test.drop(items_features, axis=1, inplace=True)


	train_itmes_features, test_itmes_features = add_items_dummies(df_train_items, df_test_items)
	train_itmes_features, test_itmes_features = drop_consumble_items(train_itmes_features, test_itmes_features)

	# Check wether hero_id of train/test have the same categories
	r_features = [f"r{i}_hero_id" for i in range(1,6)]
	d_features = [f"d{i}_hero_id" for i in range(1,6)]

	hero_id_r_train = set(pd.Series(df_train[r_features].values.flatten()).value_counts().index)
	hero_id_d_train = set(pd.Series(df_train[d_features].values.flatten()).value_counts().index)
	hero_id_r_test = set(pd.Series(df_test[r_features].values.flatten()).value_counts().index)
	hero_id_d_test = set(pd.Series(df_test[d_features].values.flatten()).value_counts().index)

	try:
		assert hero_id_r_test == hero_id_r_train
		assert hero_id_d_test == hero_id_d_train
	except AssertionError:
		print("Hero id categories are not matching between train and test.")
	else:
		print("Hero id categories are consistant across train and test.")

	game_mode_train = set(pd.Series(df_train["game_mode"].values.flatten()).value_counts().index)
	game_mode_test = set(pd.Series(df_test["game_mode"].values.flatten()).value_counts().index)

	try:
		assert game_mode_test.issubset(game_mode_train)
	except AssertionError:
		print("Game mode categories are not matching between train and test.")
	else:
		print("Game mode categories are consistant across train and test.")


	builder = AttributesAdder()
	X_train = builder.fit_transform(df_train)
	X_test = builder.transform(df_test)

	print("Combining all features")
	X_train = pd.merge(X_train,train_itmes_features,how="inner",left_index=True,right_index=True)
	X_test = pd.merge(X_test,test_itmes_features,how="inner",left_index=True,right_index=True)

	print("Persisting processed features as pickle files")
	X_train.to_pickle(ROOT_DIR/"data"/"processed"/"X_train.pkl")
	X_test.to_pickle(ROOT_DIR/"data"/"processed"/"X_test.pkl")
	y_train.to_pickle(ROOT_DIR/"data"/"processed"/"y_test.pkl")

	elapsed_time = dt.datetime.now() - elapsed_time
	print("Building features took {} s".format(elapsed_time))