import pandas as pd
import vaex
from imperio import BoxCoxTransformer
from vaex.ml.datasets import load_iris
from sklearn.base import BaseEstimator, TransformerMixin
from goldilox import Pipeline


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


def test_imperio_skleran():
    df = load_iris().to_pandas_df()
    columns = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']
    target = 'class_'

    # Imperio works differently on DataFrames and Numpy arrays
    class PandasTransformer(BoxCoxTransformer):
        def transform(self, X):
            if isinstance(X, pd.DataFrame):
                results = BoxCoxTransformer.transform(self, X.values)
                columns = list(X.columns)
                ret = pd.DataFrame(results, columns=columns)
                if y is not None and y not in ret:
                    ret[y.name] = y
                # if y is not None
                return ret
            print('numpy')
            return BoxCoxTransformer.transform(self, X.values)
    PandasTransformer().fit(df[columns],df[target]).transform(df.head(1))


    pipeline = Pipeline.from_sklearn(PandasTransformer()).fit(df[columns], df[target])

    raw = df.head(1).to_dict(orient='reocords')[0]
    pipeline.inference(raw)

    assert pipeline.validate()
    assert pipeline.inference(pipeline.raw).shape