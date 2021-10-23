from goldilox import Pipeline
def test_lightgbm():
    import vaex
    import numpy as np
    from vaex.ml.lightgbm import LightGBMModel
    from sklearn.metrics import accuracy_score
    import json

    train, test = vaex.ml.datasets.load_iris_1e5().ml.train_test_split(test_size=0.2, verbose=False)
    features = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']
    target = 'class_'

    booster = LightGBMModel(features=features,
                            target=target,
                            prediction_name='lgm_predictions',
                            num_boost_round=500, params={'verbose': -1,
                                                         'objective': 'multiclass',
                                                         'num_class': 3})
    booster.fit(train)
    train = booster.transform(train)

    @vaex.register_function()
    def argmax(ar, axis=1):
        return np.argmax(ar, axis=axis)

    train.add_function('argmax', argmax)
    train['prediction'] = train['lgm_predictions'].argmax()

    pipeline = Pipeline.from_dataframe(train)
    pipeline.set_variable('accuracy',
                          accuracy_score(pipeline.inference(test[features])['prediction'].values, test[target].values))
    data = test.to_records(0)
    pipeline.inference(test).head(1)