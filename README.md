# SC1015 DS Project - *Stroke Prediction*
## Problem Definition
- Positive stroke events are a minority making up only `4.87%` of the entire dataset. 
- So, is having a stroke an anomaly? Can it be predicted using the other variables?

## Notebooks
The code should be viewed in this order (kindly *`Trust`* the notebook if opening in JupyterNotebook)
1. [Notebook 1](https://github.com/Armaan-Goel-NTU/SC1015-DS-Project/blob/main/Notebooks/Notebook_1.ipynb) - Problem Definition, Data Cleaning, EDA, Data Preparation for ML Analysis
2. [Notebook 2](https://github.com/Armaan-Goel-NTU/SC1015-DS-Project/blob/main/Notebooks/Notebook_2.ipynb) - Isolation Forest
3. [Notebook 3](https://github.com/Armaan-Goel-NTU/SC1015-DS-Project/blob/main/Notebooks/Notebook_3.ipynb) - Random Forest and Logistic Regression

## Datasets
1. [healthcare-dataset-stroke-data.csv](https://github.com/Armaan-Goel-NTU/SC1015-DS-Project/blob/main/Datasets/healthcare-dataset-stroke-data.csv) - Original Dataset, retrieved from [kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
2. [mldata.csv](https://github.com/Armaan-Goel-NTU/SC1015-DS-Project/blob/main/Datasets/mldata.csv) - Cleaned Dataset, Prepared for ML Analysis

## ML Models Used 
1. Isolation Forest
2. Random Forest
3. Logistic Regression

## Conclusion
- `IsolationForest` suggests that **Yes**, having a stroke is indeed an anomaly.
- `RandomForest` and `LogisticRegression` performed similarly (and equally bad) at predicting stroke events. 
- Both reacted negatively to a class imbalance and had similar confusion matrices after downsampling.
- There is still more meaningful analysis that can be done. One example is using clustering models to identify at-risk groups for stroke awareness.

## New Learning 
- Using `Point-biserial correlation`, `Matthew's Correlation Coefficient`, `Cramer's V` for comparing dichotomous categorical variables to other variables. 
- Using unsupervised learning for anomaly detection.
- Concepts about `contamination` and `anomaly scores` in `IsolationForest` and `Permutation Feature Importance` in `RandomForest`. 
- Using `RandomForest` and `LogisticRegression` from sklearn.
- Using `RandomizedSearchCV` and `GridSearchCV` for hyperparameter tuning.
- Live Collaboration using `JupyterLab` in `--collaborative` mode.

## Contributions
- **Armaan** - Notebook 1
- **Lavanya** - Notebook 2
- **Delaney** - Notebook 3

## References
- https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
- https://stackoverflow.com/questions/20892799/using-pandas-calculate-cram%C3%A9rs-coefficient-matrix
- https://en.wikipedia.org/wiki/Phi_coefficient
- https://towardsdatascience.com/point-biserial-correlation-with-python-f7cd591bd3b1
- https://towardsdatascience.com/outlier-detection-with-isolation-forest-3d190448d45e
- https://www.youtube.com/watch?v=TP3wdwD8JVY
- https://mljar.com/blog/feature-importance-in-random-forest/
- https://www.analyticsvidhya.com/blog/2021/06/understanding-random-forest/
- https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
- https://www.analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/ 
- https://www.linkedin.com/pulse/building-logistic-regression-model-roc-curve-abu
- https://medium.com/analytics-vidhya/how-to-improve-logistic-regression-b956e72f4492
- https://www.analyticsvidhya.com/blog/2021/06/tune-hyperparameters-with-gridsearchcv/
