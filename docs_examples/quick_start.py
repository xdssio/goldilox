# Vaex
import numpy as np
from vaex.ml.datasets import load_iris
from vaex.ml.xgboost import XGBoostModel

df = load_iris()

# feature engineering example
df["petal_ratio"] = df["petal_length"] / df["petal_width"]

# modeling
booster = XGBoostModel(
    features=['petal_length', 'petal_width',
              'sepal_length', 'sepal_width', 'petal_ratio'],
    target='class_',
    prediction_name="prediction",
    num_boost_round=500,
)
booster.fit(df)
df = booster.transform(df)

# post modeling procssing example
df['prediction'] = np.around(df['prediction'])
df["label"] = df["prediction"].map({0: "setosa", 1: "versicolor", 2: "virginica"})

# sklearn

import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn.datasets import load_iris

# Get teh data
iris = load_iris()
features = iris.feature_names
df = pd.DataFrame(iris.data, columns=features)
df['target'] = iris.target

model = XGBClassifier().fit(df[features], df['target'])

from goldilox import Pipeline

# vaex
pipeline = Pipeline.from_vaex(df)

# sklearn - When using sklearn, we want to have a data example
#                              of the raw production query data
pipeline = Pipeline.from_sklearn(model, raw=Pipeline.to_raw(df[features]))

# Save and load
# pipeline.save( < path >)
# pipeline = Pipeline.from_file( < path >)
