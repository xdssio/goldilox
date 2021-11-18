from numbers import Number

import numpy as np
import vaex
from river import compose
from river.linear_model import LogisticRegression
from river.metrics import Accuracy
from river.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from vaex.ml.datasets import load_titanic

from goldilox import Pipeline


def test_river_vaex():
    df = load_titanic()
    features = df.get_column_names()
    features.remove('survived')
    num = compose.SelectType(Number) | StandardScaler()
    cat = compose.SelectType(str) | OneHotEncoder()
    model = (num + cat) | LogisticRegression()

    metric = Accuracy()
    for x in df.to_records():
        y = bool(x.pop('survived'))
        y_pred = model.predict_one(x)
        metric = metric.update(y, y_pred)
        model = model.learn_one(x, y)

    model.predict_one(x)

    @vaex.register_function(on_expression=False)
    def predict(*columns):
        batch = np.array(columns).T
        return np.array(
            [model.predict_one({feature: value for feature, value in zip(values, features)}) for values in batch])

    df.add_function('predict', predict)
    df['predictions'] = df.func.predict(*tuple([df[col] for col in features]))
    pipeline = Pipeline.from_vaex(df)
    pipeline.validate()
    assert pipeline.inference(pipeline.raw).shape == (1, 15)


def test_river_sklearn():
    df = load_titanic().to_pandas_df()
    features = list(df.columns)
    features.remove('survived')
    num = compose.SelectType(Number) | StandardScaler()
    cat = compose.SelectType(str) | OneHotEncoder()
    model = (num + cat) | LogisticRegression()

    class RiverLogisticRegression(BaseEstimator, TransformerMixin):

        def __init__(self, model, target, output_column='predictions'):
            self.model = model
            self.target = target
            self.metric = Accuracy()
            self.output_column = output_column

        def iterate(self, X, y):
            if y is not None:
                X = X.drop(self.target, errors='ignore')
                return zip(X, y)
            for x in df.to_dict(orient='records'):
                y = x.pop(self.target, None)
                yield x, y

        def fit(self, X, y=None, **kwargs):
            for x, y in self.iterate(X, y):
                y_pred = self.model.predict_one(x)
                self.metric.update(y, y_pred)
                self.model.learn_one(x, y)
            return self

        def predict(self, X):
            return np.array([self.model.predict_one(x) for x in X.to_dict(orient='records')])

        def transform(self, X):
            X = X.drop(self.target, errors='ignore')
            X[self.output_column] = self.predict(X)
            return X

        def fit_transform(self, X, y=None, **fit_params):
            self.fit(X, y)
            return self.transform(X)

    pipeline = Pipeline.from_sklearn(RiverLogisticRegression(model, 'survived')).fit(df)

    pipeline.validate()
    assert pipeline.inference(pipeline.raw).shape == (1, 15)
