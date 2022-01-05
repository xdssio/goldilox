# SKlearn

import pandas as pd
from sklearn.datasets import load_iris

# Get teh data
iris = load_iris()
features = iris.feature_names
df = pd.DataFrame(iris.data, columns=features)
df['target'] = iris.target

df.head()

from goldilox import Pipeline
from lightgbm.sklearn import LGBMClassifier
import json

X, y = df[features], df['target']
model = LGBMClassifier()

pipeline = Pipeline.from_sklearn(model).fit(X, y)

# I/O Example
print(f"predict for {json.dumps(pipeline.raw, indent=4)}")
pipeline.inference(pipeline.raw)

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline

# Option1: We create a Goldilox Pipeline, and running fit()
imputer = ColumnTransformer([('features_mean', SimpleImputer(strategy='mean'), features)], remainder='passthrough')
skleran_pipeline = SklearnPipeline([('imputer', imputer), ('classifier', LGBMClassifier())])
pipeline = Pipeline.from_sklearn(skleran_pipeline).fit(X, y)

# Option2: We first train the skleran.pipeline.Pipeline, and then use it + a raw example.
skleran_pipeline = SklearnPipeline([('imputer', imputer), ('classifier', LGBMClassifier())]).fit(X, y)
pipeline = Pipeline.from_sklearn(skleran_pipeline, raw=raw)  # <--- provide the raw

# Vaex
import vaex
import warnings
from vaex.ml.datasets import load_iris
from vaex.ml.lightgbm import LightGBMModel
from goldilox import Pipeline
import numpy as np
import json

warnings.filterwarnings('ignore')

df = load_iris()
target = 'class_'

# feature engineering example
df['petal_ratio'] = df['petal_length'] / df['petal_width']

booster = LightGBMModel(features=['petal_length', 'petal_width', 'sepal_length', 'sepal_width', 'petal_ratio'],
                        target=target,
                        prediction_name='predictions',
                        num_boost_round=500, params={'verbosity': -1,
                                                     'objective': 'multiclass',
                                                     'num_class': 3})
booster.fit(df)
df = booster.transform(df)


# post model processing example
@vaex.register_function()
def argmax(ar, axis=1):
    return np.argmax(ar, axis=axis)


df.add_function('argmax', argmax)
df['prediction'] = df['predictions'].argmax()

df['label'] = df['prediction'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Vaex remember all the transformations, this is a skleran.pipeline alternative
pipeline = Pipeline.from_vaex(df, description='simple lightGBM')
pipeline.raw.pop(target)  # (optional) we don't expect to get the class_ in queries

# I/O Example
print(f"predict for {json.dumps(pipeline.raw, indent=4)}")
pipeline.inference(pipeline.raw)

from vaex.ml.datasets import load_iris
from vaex.ml.sklearn import Predictor
from lightgbm.sklearn import LGBMClassifier

df = load_iris()
target = 'class_'

# feature engineering example
df['petal_ratio'] = df['petal_length'] / df['petal_width']
features = ['petal_length', 'petal_width', 'sepal_length',
            'sepal_width', 'petal_ratio']
model = Predictor(model=LGBMClassifier(**{'verbosity': -1,
                                          'objective': 'multiclass',
                                          'num_class': 3}),
                  features=features,
                  target=target,
                  prediction_name='prediction')
model.fit(df)
df = model.transform(df)
df['label'] = df['prediction'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
df.head(2)
