import pandas as pd
from sklearn.datasets import load_iris

# Get teh data
iris = load_iris()
features = iris.feature_names
df = pd.DataFrame(iris.data, columns=features)
df['target'] = iris.target

df.head(2)

from goldilox import Pipeline
from xgboost.sklearn import XGBClassifier
import json

X, y = df[features], df['target']
model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")

pipeline = Pipeline.from_sklearn(model).fit(X, y)

# I/O Example
raw = pipeline.raw
print(f"predict for {json.dumps(raw, indent=4)}")
pipeline.inference(raw)

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline

# Option1: We create a Goldilox Pipeline, and running fit()
imputer = ColumnTransformer([('features_mean', SimpleImputer(strategy='mean'),
                              features)], remainder='passthrough')
skleran_pipeline = SklearnPipeline([('imputer', imputer),
                                    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric="mlogloss"))])
pipeline = Pipeline.from_sklearn(skleran_pipeline).fit(X, y)

# Option2: We first train the skleran.pipeline.Pipeline, and then use it + a raw example.
skleran_pipeline = SklearnPipeline(
    [('imputer', imputer), ('classifier', XGBClassifier(use_label_encoder=False, eval_metric="mlogloss"))]).fit(X, y)
pipeline = Pipeline.from_sklearn(skleran_pipeline, raw=raw)  # <--- provide the raw

# Vaex
import json
import warnings
from vaex.ml.datasets import load_iris
from vaex.ml.xgboost import XGBoostModel
from goldilox import Pipeline

warnings.filterwarnings('ignore')

df = load_iris()
target = 'class_'

# feature engineering example
df['petal_ratio'] = df['petal_length'] / df['petal_width']

booster = XGBoostModel(
    params={"eval_metric": "mlogloss",
            "objective": "multi:softmax",
            "num_class": df[target].nunique()},
    features=[
        "petal_length",
        "petal_width",
        "sepal_length",
        "sepal_width",
        "petal_ratio",
    ],
    target=target,
    prediction_name="prediction",
    num_boost_round=500,
)
booster.fit(df)
df = booster.transform(df)

df['label'] = df['prediction'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Vaex remember all the transformations, this is a skleran.pipeline alternative
pipeline = Pipeline.from_vaex(df, description='simple lightGBM')
pipeline.raw.pop(target)  # (optional) we don't expect to get the class_ in queries

# I/O Example
print(f"predict for {json.dumps(pipeline.raw, indent=4)}")
pipeline.inference(pipeline.raw)

# vaex + sklearn
from vaex.ml.datasets import load_iris
from vaex.ml.sklearn import Predictor
from xgboost.sklearn import XGBClassifier

df = load_iris()
target = 'class_'

# feature engineering example
df['petal_ratio'] = df['petal_length'] / df['petal_width']
features = ['petal_length', 'petal_width', 'sepal_length',
            'sepal_width', 'petal_ratio']
model = Predictor(model=XGBClassifier(**{"eval_metric": "mlogloss",
                                         "objective": "multi:softmax",
                                         "num_class": df[target].nunique()}),
                  features=features,
                  target=target,
                  prediction_name='prediction')
model.fit(df)
df = model.transform(df)
df['label'] = df['prediction'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

from goldilox import Pipeline

pipeline = Pipeline.from_vaex(df)

print(pipeline.inference(pipeline.raw))
