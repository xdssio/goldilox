import sklearn.pipeline
import vaex
from sklearn.linear_model import LogisticRegression
from vaex.ml.sklearn import Predictor

from goldilox.datasets import load_iris
from goldilox.sklearn.pipeline import SklearnPipeline
from goldilox.vaex.pipeline import VaexPipeline
from tests.test_utils import validate_persistence


def test_skleran_save_load(tmpdir):
    df, features, target = load_iris()
    X, y = df[features], df[target]
    self = pipeline = SklearnPipeline.from_sklearn(sklearn.pipeline.Pipeline([('regression', LogisticRegression())]),
                                                   output_columns=['predictiom'])
    pipeline.fit(X, y)
    pipeline = validate_persistence(pipeline)
    assert pipeline.inference(pipeline.raw).shape == (1, 5)


def test_vaex_save_load(tmpdir):
    df, features, target = load_iris()
    df = vaex.from_pandas(df)
    X, y = df[features], df[target]

    model = Predictor(model=LogisticRegression(), features=features, target=target)
    model.fit(df)
    df = model.transform(df)
    pipeline = VaexPipeline.from_vaex(df)
    pipeline = validate_persistence(pipeline)
    assert pipeline.inference(pipeline.raw).shape == (1, 6)


def test_goldilox_save_load(tmpdir):
    df, features, target = load_iris()
    df = vaex.from_pandas(df)
    model = Predictor(model=LogisticRegression(), features=features, target=target)
    model.fit(df)
    df = model.transform(df)
    pipeline = VaexPipeline.from_vaex(df)
    pipeline = validate_persistence(pipeline)
    assert pipeline.pipeline_type == 'vaex'
    pipeline.inference(pipeline.raw).shape == (1, 6)

    df = df.to_pandas_df()
    X = df[features]
    y = df[target]
    pipeline = SklearnPipeline.from_sklearn(sklearn.pipeline.Pipeline([('regression', LogisticRegression())])).fit(X, y)
    pipeline = validate_persistence(pipeline)
    assert pipeline.pipeline_type == 'sklearn'
    assert pipeline.inference(pipeline.raw).shape == (1, 5)
