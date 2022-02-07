from tempfile import TemporaryDirectory

import numpy as np
import vaex

from goldilox import Pipeline
from goldilox.datasets import load_iris


def test_mlflow():
    df, features, target = load_iris()
    df = vaex.from_pandas(df)

    from vaex.ml.lightgbm import LightGBMModel

    booster = LightGBMModel(features=features,
                            target=target,
                            prediction_name='lgbm',
                            num_boost_round=500,
                            params={'verbosity': -1,
                                    'objective': 'multiclass',
                                    'num_class': 3})
    booster.fit(df)
    df = booster.transform(df)

    @vaex.register_function()
    def argmax(ar, axis=1):
        return np.argmax(ar, axis=axis)

    df.add_function('argmax', argmax)
    df['prediction'] = df['lgbm'].argmax()
    pipeline = Pipeline.from_vaex(df)

    path = str(TemporaryDirectory().name) + '/pipeline'
    pipeline.save(path, mlflow=True)

    """mlflow models serve -m <path> --no-conda"""
    import mlflow.pyfunc
    model = mlflow.pyfunc.load_model(path)
    assert len(model.predict(pipeline.to_pandas(pipeline.raw))) == 1
