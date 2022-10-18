import numpy as np
import pytest
import vaex
from vaex.ml.lightgbm import LightGBMModel
from vaex.ml.sklearn import Predictor
from xgboost.sklearn import XGBClassifier

from goldilox.datasets import load_iris
from goldilox.vaex.pipeline import VaexPipeline as VaexPipeline


@pytest.fixture()
def iris():
    # iris = load_iris()
    return load_iris()


def test_vaex_cross_validate(iris):
    df, features, target = load_iris()
    df = vaex.from_pandas(df)

    def fit(df):
        df['petal_ratio'] = df['petal_length'] / df['petal_width']

        xgb = Predictor(model=XGBClassifier(use_label_encoder=False), target=target, features=features,
                        prediction_name='predictions')
        xgb.fit(df)
        df = xgb.transform(df)

        # cross_validate(xgb.model, df.to_pandas_df(), df[target].values, scoring=('accuracy', 'roc_auc_ovr'), cv=3)

        df = xgb.transform(df)
        df['predictions'] = df['predictions'].astype('int')
        lgb = LightGBMModel(features=features,
                            target=target,
                            prediction_name='probabilities',
                            num_boost_round=500, params={'verbose': -1,
                                                         'objective': 'multiclass',
                                                         'num_class': 3})
        lgb.fit(df)
        df = lgb.transform(df)
        df.variables['variables'] = {'predict_column': 'predictions', 'predict_proba_column': 'probabilities',
                                     'target_column': 'target'}
        return df

    self = pipeline = VaexPipeline.from_dataframe(df, fit=fit, target=target).fit(df)
    self.variables = {'predict_column': 'predictions', 'predict_proba_column': 'probabilities', 'target_column': target}

    pipeline.cross_validate(df.to_pandas_df(), df[target].values, scoring=('accuracy', 'roc_auc_ovr'), cv=3,
                            error_score=np.nan)

    from sklearn.metrics import accuracy_score, make_scorer
    scorer = make_scorer(accuracy_score)
    scorer(self, df, df[target].values)
