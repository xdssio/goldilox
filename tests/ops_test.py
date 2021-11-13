

from goldilox.sklearn.pipeline import Pipeline as SklearnPipeline
from goldilox.vaex.pipeline import VaexPipeline as VaexPipeline
from vaex.ml.datasets import load_iris_1e5



def lightgbm_vaex_fit():
    iris = load_iris_1e5()

    def fit(df):
        import vaex
        import numpy as np
        from vaex.ml.lightgbm import LightGBMModel
        from sklearn.metrics import accuracy_score

        train, test = df.ml.train_test_split(test_size=0.2, verbose=False)

        features = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']
        target = 'class_'

        booster = LightGBMModel(features=features,
                                target=target,
                                prediction_name='predictions',
                                num_boost_round=500, params={'verbose': -1,
                                                             'objective': 'multiclass',
                                                             'num_class': 3})
        booster.fit(df)

        @vaex.register_function()
        def argmax(ar, axis=1):
            return np.argmax(ar, axis=axis)

        train = booster.transform(df)
        train.add_function('argmax', argmax)
        train['prediction'] = train['predictions'].argmax()

        pipeline = VaexPipeline.from_dataframe(train)
        accuracy = accuracy_score(pipeline.inference(test[features])['prediction'].values,
                                  test[target].values)
        booster = LightGBMModel(features=features,
                                target=target,
                                prediction_name='predictions',
                                num_boost_round=500, params={'verbose': -1,
                                                             'objective': 'multiclass',
                                                             'num_class': 3})
        booster.fit(df)
        df = booster.transform(df)
        df.add_function('argmax', argmax)
        df['prediction'] = df['predictions'].argmax()
        names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        df['label'] = df['prediction'].map(names)
        df['probabilities'] = df['predictions'].apply(lambda x: {names.get(i):x[i] for i in range(3)})
        df.variables['accuracy'] = accuracy
        df.variables['names'] = names
        return df

    df = iris.copy()
    pipeline = VaexPipeline.from_dataframe(df, fit=fit, description="Lightgbm with Vaex")
    data = df.to_records(0)
    assert pipeline.inference(data).shape == df.head(1).shape
    pipeline.fit(df)

    assert pipeline.inference(data).shape == (1, 9)
    assert pipeline.get_variable('accuracy')
    assert pipeline.get_variable('names')
    assert pipeline.raw == data
    assert list(pipeline.example.keys()) == ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class_',
                                             'predictions', 'prediction','label','probabilities']
    assert 'Lightgbm' in pipeline.description
    pipeline.raw.pop('class_')
    assert pipeline.inference(data).shape == (1, 9)
    pipeline.validate(df.head(2))

    pipeline.save('../goldilox-ops/models/pipeline.pkl')


def test_lightgbm_sklearn():
    from lightgbm.sklearn import LGBMClassifier
    import sklearn.pipeline

    df = load_iris_1e5().copy()
    features = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']
    target = 'class_'
    sk_pipeline = sklearn.pipeline.Pipeline([('classifier', LGBMClassifier())])
    X = df[features]
    y = df[target]
    self = pipeline = SklearnPipeline.from_sklearn(sk_pipeline, description="Lightgbm with sklearn").fit(X, y)

    assert pipeline.inference(X).head(10).shape == (10, 5)
    assert pipeline.inference(X.values[:10]).shape == (10, 5)
    assert pipeline.inference(self.raw).shape == (1, 5)

    pipeline.fit(df)
    assert pipeline.inference(X).head(10).shape == (10, 5)
    assert pipeline.inference(X.values[:10]).shape == (10, 5)
    assert pipeline.inference(self.raw).shape == (1, 5)

    # with a trained sklearn pipeline
    sample = X.head(1).to_records()[0]
    self = pipeline = SklearnPipeline.from_sklearn(sk_pipeline, raw=sample, description="Lightgbm with sklearn").fit(X, y)
    assert pipeline.inference(X).head(10).shape == (10, 5)
    assert pipeline.inference(X.values[:10]).shape == (10, 5)
    assert pipeline.inference(self.raw).shape == (1, 5)
    assert 'Lightgbm' in pipeline.description
    pipeline.save('../goldilox-ops/models/sk.pkl')
