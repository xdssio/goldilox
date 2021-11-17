from numbers import Number

import numpy as np
import vaex
from river import compose
from river import metrics
from river.linear_model import LogisticRegression
from river.preprocessing import StandardScaler, OneHotEncoder
from vaex.ml.datasets import load_titanic

from goldilox import Pipeline


def test_river_vaex():
    df = load_titanic()
    features = df.get_column_names()
    features.remove('survived')
    num = compose.SelectType(Number) | StandardScaler()
    cat = compose.SelectType(str) | OneHotEncoder()
    model = (num + cat) | LogisticRegression()

    metric = metrics.Accuracy()
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
