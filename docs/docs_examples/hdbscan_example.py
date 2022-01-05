import pandas as pd
from sklearn.datasets import make_blobs

n_features = 10
features = [f"f{i}" for i in range(n_features)]
blobs, labels = make_blobs(n_samples=2000, n_features=n_features)
df = pd.DataFrame(blobs, columns=features)
df['target'] = labels
df.head(2)

# vaex
from hdbscan import HDBSCAN
from goldilox import Pipeline
import numpy as np
import vaex

df = vaex.from_pandas(df)
model = HDBSCAN(prediction_data=True)
model.fit(df[features], df['target'])


@vaex.register_function()
def hdbscan(*columns):
    data = np.array(columns).T
    labels, _ = approximate_predict(model, data)
    return labels


df.add_function('hdbscan', hdbscan)
df['label'] = df.func.hdbscan(*features)

pipeline = Pipeline.from_vaex(df)
pipeline.inference(pipeline.raw)

# sklearn
from sklearn.base import TransformerMixin, BaseEstimator
from hdbscan import HDBSCAN, approximate_predict
from goldilox import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import sklearn.pipeline

X, y = df[features], df['target']


class HDBSCANTransformer(TransformerMixin, BaseEstimator):

    def __init__(self, prediction_column='label', **kwargs):
        kwargs['prediction_data'] = True
        self.model = HDBSCAN(**kwargs)
        self.prediction_column = prediction_column

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        labels, strength = approximate_predict(self.model, X)
        return labels

    def transform(self, X):
        X[self.prediction_column] = self.predict(X)
        return X


imputer = ColumnTransformer([('features_mean', SimpleImputer(strategy='mean'), features)], remainder='passthrough')
skleran_pipeline = sklearn.pipeline.Pipeline([('imputer', imputer), ('classifier', HDBSCANTransformer())])
pipeline = Pipeline.from_sklearn(skleran_pipeline).fit(X, y)
pipeline.inference(pipeline.raw)
