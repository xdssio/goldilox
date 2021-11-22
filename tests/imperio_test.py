import pandas as pd
import vaex
from imperio import BoxCoxTransformer
from vaex.ml.datasets import load_iris
from sklearn.base import BaseEstimator, TransformerMixin
from goldilox import Pipeline
import pytest

def test_imperio_vaex():
    df = load_iris()
    columns = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']
    target = 'class_'
    tr = BoxCoxTransformer().fit(df[columns], df[target])

    @vaex.register_function()
    def box_cox(ar):
        column, target = 'column', 'target'
        batch = pd.DataFrame({column: ar, target: ar})
        return tr.apply(batch, 'target', [column])[column].values

    df.add_function('box_cox', box_cox)
    for column in columns:
        df[f"box_cox_{column}"] = df[column].box_cox()

    pipeline = Pipeline.from_vaex(df)
    assert pipeline.validate()
    assert pipeline.inference(pipeline.raw).shape == (1, 9)

@pytest.mark.skip("TODO")
def test_imperio_skleran():
    df = load_iris().to_pandas_df()
    columns = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']
    target = 'class_'

    # Imperio works differently on DataFrames and Numpy arrays
    class PandasTransformer(BoxCoxTransformer):

        def __init__(self, features, target, **kwargs):
            self.target = target
            self.features = features
            super().__init__(self, **kwargs)

        def transform(self, X, **kwargs):
            if isinstance(X, pd.DataFrame):

                return self.apply(X, self.target, self.features)
            else:
                print(X[0])
                return BoxCoxTransformer.transform(self, X=X, **kwargs)
    PandasTransformer(features=columns, target='class_').fit(df[columns], df[target]).transform(df)
    pipeline = Pipeline.from_sklearn(PandasTransformer(features=columns, target='class_')).fit(df[columns], df[target])
    raw = df.head(1).to_dict(orient='records')[0]
    pipeline.inference(raw)

    assert pipeline.validate()
    assert pipeline.inference(pipeline.raw).shape