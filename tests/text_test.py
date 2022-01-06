import pytest
import sklearn.pipeline
import vaex
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from goldilox.sklearn.pipeline import Pipeline as SklearnPipeline
from goldilox.vaex.pipeline import VaexPipeline as VaexPipeline
from tests.test_utils import validate_persistence


@pytest.fixture()
def news():
    # df = news = vaex.open('data/news.hdf5').head(1000)
    return vaex.open('data/news.hdf5').head(1000)


def test_text_vaex(news, tmpdir):
    df = news.copy()
    sk_pipeline = sklearn.pipeline.Pipeline([('tfidf', TfidfVectorizer()), ('classifier', LogisticRegression())])
    X = df['text'].to_numpy()
    y = df['target'].to_numpy()
    sk_pipeline.fit(X, y)

    @vaex.register_function()
    def predict(ar):
        return sk_pipeline.predict(ar.tolist())

    df.add_function('predict', predict)
    df['prediction'] = df['text'].predict()
    pipeline = VaexPipeline.from_dataframe(df)
    pipeline.raw.pop('target')
    path = str(tmpdir) + '/model.pkl'
    pipeline.save(path)
    pipeline = SklearnPipeline.from_file(path)
    assert pipeline.inference(pipeline.raw).shape == (1, 3)


def test_text_sklearn(news, tmpdir):
    df = news.copy()

    sk_pipeline = sklearn.pipeline.Pipeline([('tfidf', TfidfVectorizer()), ('classifier', LogisticRegression())])
    pipeline = SklearnPipeline.from_sklearn(sk_pipeline, features=['text'], target='target').fit(df)
    assert pipeline.inference(pipeline.raw).shape == (1, 2)
    df = df.to_pandas_df()
    X = df['text']
    y = df['target']
    sk_pipeline = sklearn.pipeline.Pipeline([('tfidf', TfidfVectorizer()), ('classifier', LogisticRegression())])
    pipeline = SklearnPipeline.from_sklearn(sk_pipeline).fit(X, y)
    pipeline = validate_persistence(pipeline)
    assert pipeline.inference(pipeline.raw).shape == (1, 2)
