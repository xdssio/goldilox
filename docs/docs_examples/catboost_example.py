import pandas as pd
from sklearn.datasets import load_iris

# Get teh data
iris = load_iris()
features = iris.feature_names
df = pd.DataFrame(iris.data, columns=features)
df['target'] = iris.target

df.head(2)

from goldilox import Pipeline
from catboost import CatBoostClassifier
import json

X, y = df[features], df['target']

model = CatBoostClassifier(verbose=0)
pipeline = Pipeline.from_sklearn(model).fit(X, y)

# I/O Example
raw = pipeline.raw
print(f"predict for {json.dumps(raw, indent=4)}")
pipeline.inference(raw)

from goldilox import Pipeline
from catboost import CatBoostClassifier
import json

X, y = df[features], df['target']

model = CatBoostClassifier(verbose=0)
pipeline = Pipeline.from_sklearn(model).fit(X, y)

# I/O Example
print(f"predict for {json.dumps(pipeline.raw, indent=4)}")
pipeline.inference(pipeline.raw)

# Vaex example
import vaex
import warnings
from vaex.ml.catboost import CatBoostModel
from goldilox import Pipeline
import numpy as np
import json

warnings.filterwarnings("ignore")

df = load_iris()
target = "class_"

# feature engineering example
df["petal_ratio"] = df["petal_length"] / df["petal_width"]

# modeling
booster = CatBoostModel(features=["petal_length", "petal_width", "sepal_length", "sepal_width", "petal_ratio"],
                        target=target,
                        prediction_name="predictions",
                        params={"num_boost_round": 500, "verbose": 0, "objective": "MultiClass"})

booster.fit(df)
df = booster.transform(df)


# post processing
@vaex.register_function()
def argmax(ar, axis=1):
    return np.argmax(ar, axis=axis)


df.add_function("argmax", argmax)
df["prediction"] = df["predictions"].argmax()

df["label"] = df["prediction"].map({0: "setosa", 1: "versicolor", 2: "virginica"})

# Vaex remember all the transformations
pipeline = Pipeline.from_vaex(df)
pipeline.raw.pop(target)  # (optional) we don't expect to get the class_ in queries

# I/O Example
print(f"predict for {json.dumps(pipeline.raw, indent=4)}")
pipeline.inference(pipeline.raw)

# vaex + skleran
from catboost import CatBoostClassifier
from vaex.ml.datasets import load_iris
from vaex.ml.sklearn import Predictor

df = load_iris()
target = 'class_'

# feature engineering example
df['petal_ratio'] = df['petal_length'] / df['petal_width']
features = ['petal_length', 'petal_width', 'sepal_length',
            'sepal_width', 'petal_ratio']
model = Predictor(model=CatBoostClassifier(verbose=0),
                  features=features,
                  target=target,
                  prediction_name='prediction')
model.fit(df)
df = model.transform(df)
df['prediction'] = df['prediction'].apply(lambda x: x[0])  # catboost returns array - so we adjust
df['label'] = df['prediction'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

