# kaggle-competition-pipeline
This is an exemple of a full data pipeline used as a template for several ML competitions such as Kaggle. The objective is to go from raw data to submissions seamlessy and to iterate fast to experiment many ideas.

The project can be structured as follows:
```bash
.
├── data
│   ├── processed
│   │   ├── df_test_raw.pkl
│   │   ├── df_train_raw.pkl
│   │   ├── X_test.pkl
│   │   ├── X_train.pkl
│   │   └── y_test.pkl
│   └── raw
├── meta_data
│   └── public_LB
│       └── mlcourse-dota2-win-prediction-publicleaderboard.csv
├── notebooks
│   └── EDA.ipynb
├── requirements.txt
├── src
│   ├── build_features.py
│   ├── ensemble.py
│   ├── train.py
│   └── utils.py
├── structure.txt
├── submissions
│   ├── submission_2019-11-18_21_06_15_LGBM.csv
│   ├── submission_2019-11-18_21_10_28_ENSEMBLE.csv
│   ├── submission_2019-11-18_21_42_29_LR.csv
└── submissions_history.ods
```

The competition was proposed as inclass assignement during the last session of https://mlcourse.ai/.

https://www.kaggle.com/c/mlcourse-dota2-win-prediction

The order of running the script is:
1 - Generate features with extensive feature engineering as export the data set as pickle object

```bash
python build_features.py root_dir ROOT_DIR
```

2- Train out of the box ML models to calibrate the complexity of the task
```bash
python train.py root_dir ROOT_DIR
```

3- Ensemble promising models based on public leaderboard scores

```bash
python ensemble.py root_dir ROOT_DIR
```

Happy kaggling :)
