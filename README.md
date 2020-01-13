# kaggle-competition-pipeline
This is an exemple of a full data pipeline used as a template for several ML competitions such as Kaggle. The objective is to go from raw data to submissions seamlessy and to iterate fast to experiment many ideas.

The competition was proposed as inclass assignement during the last session of https://mlcourse.ai/.

https://www.kaggle.com/c/mlcourse-dota2-win-prediction

The order of running the script is:
1 - Generate features with extensive feature engineering as export the data set as pickle object

```bash
python build_features.py root_dir "."
```

2- Train out of the box ML models to calibrate the complexity of the task
```bash
python train.py root_dir "."
```

3- Ensemble promising models based on public leaderboard scores

```bash
python ensemble.py root_dir "."
``

Happy kaggling :)
