import numpy as np
from vaex.ml.datasets import load_iris
from vaex.ml.xgboost import XGBoostModel

df = load_iris()
df.head(2)

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
df.head(2)

from goldilox import Pipeline

pipeline = Pipeline.from_vaex(df)
print(pipeline.inference(pipeline.raw))

from vaex.ml.datasets import load_iris


def fit(df):
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
    df["label"] = df["prediction"].map({0: "setosa",
                                        1: "versicolor",
                                        2: "virginica"})
    return df


df = load_iris()
pipeline = pipeline.from_vaex(df, fit=fit)
pipeline.fit(df)
print(pipeline.inference(pipeline.raw))
