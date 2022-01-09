# vaex
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
df["label"] = df["prediction"].map({
    0: "setosa",
    1: "versicolor",
    2: "virginica"})
df.head(2)

from goldilox import Pipeline

pipeline = Pipeline.from_vaex(df)

pipeline.inference({'sepal_length': 5.9,
                    'sepal_width': 3.0,
                    'petal_length': 4.2,
                    'petal_width': 1.5})
pipeline.inference(df)

from vaex.ml.datasets import load_iris
from goldilox import Pipeline

df = load_iris()
df_class_1 = df[df['class_'] == 1]
pipeline = Pipeline.from_vaex(df_class_1)
pipeline.inference(df, set_filter=False).tail(2)['class_']
pipeline.inference(df, set_filter=True).tail(2)['class_']

from vaex.ml.datasets import load_iris
from goldilox import Pipeline

df = load_iris()
pipeline = Pipeline.from_vaex(df_class_1)
pipeline.inference({'sepal_length': 5.9,
                    'sepal_width': 3.0,
                    'petal_length': 4.2,
                    'petal_width': 1.5,
                    'extra_column': "stuff"})

pipeline.inference({'sepal_length': 5.9,
                    'sepal_width': 3.0,
                    'petal_length': 4.2,
                    'petal_width': 1.5,
                    'extra_column': "stuff"},
                   passthrough=False)
