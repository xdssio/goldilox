import numpy as np
import pytest
import sklearn.pipeline
import vaex
from sklearn.metrics import accuracy_score

from goldilox import Pipeline
from goldilox.datasets import load_iris
from goldilox.sklearn.pipeline import Pipeline as SklearnPipeline
from goldilox.vaex.pipeline import VaexPipeline as VaexPipeline
from tests.tests_utils import validate_persistence


@pytest.fixture()
def iris():
    # iris = load_iris()
    return load_iris()


def test_vaex_catboost(iris):
    df, features, target = iris
    df = vaex.from_pandas(df)
    train, test = df.ml.train_test_split(test_size=0.2, verbose=False)
    train["X"] = train["petal_length"] / train["petal_width"]
    from vaex.ml.catboost import CatBoostModel

    booster = CatBoostModel(
        features=features,
        target=target,
        task_type="GPU",
        prediction_name="predictions",
        num_boost_round=500,
        params={"verbose": 0, "iterations": 10, "objective": "MultiClass"},
    )
    booster.fit(train)
    train = booster.transform(train)

    @vaex.register_function()
    def argmax(ar, axis=1):
        return np.argmax(ar, axis=axis)

    train.add_function("argmax", argmax)
    train["prediction"] = train["predictions"].argmax()

    pipeline = VaexPipeline.from_dataframe(train)
    pipeline.set_variable(
        "accuracy",
        accuracy_score(
            pipeline.inference(test[features])["prediction"].values, test[target].values
        ),
    )

    sample = test.to_records(0)

    assert pipeline.inference(test).head(1).shape == (1, 8)
    assert pipeline.inference(sample).head(1).shape == (1, 8)
    assert pipeline.inference({"sepal_length": 5.9, "petal_length": 4.2}).head(
        1
    ).shape == (1, 8)
    assert pipeline.inference(
        {"sepal_length": 5.9, "petal_length": 4.2}, columns="prediction"
    ).head(1).shape == (1, 1)
    assert pipeline.inference(
        {"sepal_length": 5.9, "petal_length": 4.2, "Y": "new column"}
    ).head(1).shape == (1, 9)


def test_catboost_vaex_fit(iris, tmpdir):
    def fit(df):
        from sklearn.metrics import accuracy_score
        from vaex.ml.catboost import CatBoostModel

        train, test = df.ml.train_test_split(test_size=0.2, verbose=False)

        features = ["petal_length", "petal_width", "sepal_length", "sepal_width"]
        target = "target"

        booster = CatBoostModel(
            features=features,
            target=target,
            prediction_name="predictions",
            num_boost_round=500,
            params={"verbose": 0, "iterations": 10, "objective": "MultiClass"},
        )
        booster.fit(train)
        train = booster.transform(train)

        @vaex.register_function()
        def argmax(ar, axis=1):
            return np.argmax(ar, axis=axis)

        train.add_function("argmax", argmax)
        train["prediction"] = train["predictions"].argmax()
        pipeline = VaexPipeline.from_dataframe(train)
        accuracy = accuracy_score(
            pipeline.inference(test[features])["prediction"].values, test[target].values
        )
        booster = CatBoostModel(
            features=features,
            target=target,
            prediction_name="predictions",
            num_boost_round=500,
            params={"verbose": 0, "iterations": 10, "objective": "MultiClass"},
        )
        booster.fit(df)
        df = booster.transform(df)
        df.variables["accuracy"] = accuracy
        df.add_function("argmax", argmax)
        df["prediction"] = df["predictions"].argmax()
        return df

    df, features, target = iris
    df = vaex.from_pandas(df)
    pipeline = VaexPipeline.from_dataframe(df, fit=fit)
    data = df.to_records(0)
    data.pop("target")
    assert pipeline.inference(data).shape == df.head(1).shape
    data = df.to_records(0)
    pipeline.fit(df)
    pipeline = validate_persistence(pipeline)
    assert pipeline.inference(data).shape == (1, 7)
    assert pipeline.get_variable("accuracy")
    assert pipeline.raw == data
    assert list(pipeline.example.keys()) == [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "target",
        "predictions",
        "prediction",
    ]


def test_catboost_sklearn(iris):
    from catboost import CatBoostClassifier

    df, features, target = load_iris()
    X, y = df[features], df[target]

    sk_pipeline = sklearn.pipeline.Pipeline(
        [("classifier", CatBoostClassifier(verbose=0, iterations=10))]
    )
    pipeline = SklearnPipeline.from_sklearn(sk_pipeline).fit(X, y)
    assert pipeline.inference(X).head(10).shape == (10, 5)
    assert pipeline.inference(X.values[:10]).shape == (10, 5)
    assert pipeline.inference(pipeline.raw).shape == (1, 5)

    df, features, target = load_iris()
    X, y = df[features], df[target]
    sk_pipeline = sklearn.pipeline.Pipeline(
        [("classifier", CatBoostClassifier(verbose=0, iterations=10))]
    )
    pipeline = SklearnPipeline.from_sklearn(sk_pipeline).fit(X, y)

    assert pipeline.inference(X).head(10).shape == (10, 5)
    assert pipeline.inference(X.values[:10]).shape == (10, 5)
    assert pipeline.inference(pipeline.raw).shape == (1, 5)

    pipeline.fit(df)
    assert pipeline.inference(X).head(10).shape == (10, 5)
    assert pipeline.inference(X.values[:10]).shape == (10, 5)
    assert pipeline.inference(pipeline.raw).shape == (1, 5)

    # with a trained sklearn pipeline
    sample = Pipeline.to_raw(X)
    pipeline = SklearnPipeline.from_sklearn(sk_pipeline, raw=sample).fit(X, y)
    assert pipeline.inference(X).head(10).shape == (10, 5)
    assert pipeline.inference(X.values[:10]).shape == (10, 5)
    assert pipeline.inference(pipeline.raw).shape == (1, 5)
