import sklearn.pipeline
from sklearn.linear_model import LogisticRegression
from vaex.ml.datasets import load_iris
from vaex.ml.sklearn import Predictor

from goldilox.datasets import load_iris
from goldilox.sklearn.pipeline import SklearnPipeline
from goldilox.vaex.pipeline import VaexPipeline
from tests.test_utils import validate_persistence

features = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']
target = 'target'


def test_skleran_save_load(tmpdir):
    iris = load_iris()
    X = iris[features]
    y = iris[target]
    self = pipeline = SklearnPipeline.from_sklearn(sklearn.pipeline.Pipeline([('regression', LogisticRegression())]),
                                                   output_columns=['predictiom'])
    pipeline.fit(X, y)
    pipeline = validate_persistence(pipeline)
    assert pipeline.inference(pipeline.raw).shape == (1, 5)


def test_vaex_save_load(tmpdir):
    df = load_iris('vaex')

    model = Predictor(model=LogisticRegression(), features=features, target=target)
    model.fit(df)
    df = model.transform(df)
    pipeline = VaexPipeline.from_vaex(df)
    pipeline = validate_persistence(pipeline)
    assert pipeline.inference(pipeline.raw).shape == (1, 6)


def test_goldilox_save_load(tmpdir):
    df = load_iris('vaex')
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
