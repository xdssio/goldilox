import logging
import warnings

import numpy as np
import pandas as pd
import pytest
import sklearn.pipeline
from lightgbm.sklearn import LGBMClassifier
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from goldilox import Pipeline
from tests.test_utils import validate_persistence

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

from goldilox.sklearn.pipeline import SklearnPipeline
from goldilox.datasets import load_iris


@pytest.fixture()
def iris():
    # iris = load_iris()
    return load_iris()


def test_sklearn_transformer(iris):
    df, features, target = load_iris()
    X, y = df[features], df[target]
    pipeline = Pipeline.from_sklearn(
        sklearn.pipeline.Pipeline([("standard", StandardScaler())]),
    ).fit(df)

    results = pipeline.inference(df.head())
    assert results.shape == (5, 5)
    assert results['target'].dtype == float

    pipeline = Pipeline.from_sklearn(
        sklearn.pipeline.Pipeline([("standard", StandardScaler())]),
        features=features
    ).fit(df)

    results = pipeline.inference(df.head())
    assert results.shape == (5, 5)
    assert results['target'].dtype == int

    pipeline = Pipeline.from_sklearn(PCA()).fit(X, y)

    results = pipeline.inference(df.head())
    assert results.shape == (5, 5)

    pipeline = Pipeline.from_sklearn(PCA(),
                                     output_columns=[f"pca{i}" for i in range(4)]
                                     ).fit(X)

    results = pipeline.inference(df.head())
    for i in range(4):
        f"pca{i}" in results
    assert results.shape == (5, 5)
    assert isinstance(results, pd.DataFrame)

    results = pipeline.inference(df.head(), columns=["pca1", "pca2", "noise", 'target'])
    for i in range(2):
        f"pca{i}" in results
    assert results.shape == (5, 3)


def test_from_sklearn_transform_numpy(iris):
    df, features, target = load_iris()
    X, y = df[features], df[target]
    values = X.values
    self = pipeline = SklearnPipeline.from_sklearn(
        sklearn.pipeline.Pipeline([("standard", StandardScaler())]),
        features=features,
        output_columns=features
    )
    pipeline.fit(values)
    assert pipeline.inference(X).shape == X.shape
    assert pipeline.inference(values).shape == X.shape
    assert pipeline.raw

    pipeline = SklearnPipeline.from_sklearn(
        sklearn.pipeline.Pipeline([("standard", StandardScaler())])
    ).fit(X)
    assert pipeline.inference(X).shape == X.shape
    assert pipeline.inference(values).shape == X.shape
    assert pipeline.raw == pipeline.to_raw(X)
    pipeline = SklearnPipeline.from_sklearn(
        sklearn.pipeline.Pipeline([("standard", StandardScaler())]), features=X.columns
    ).fit(X)
    assert pipeline.inference(X).shape == X.shape
    assert pipeline.inference(values).shape == X.shape
    assert pipeline.raw == pipeline.to_raw(X)

    pipeline = SklearnPipeline.from_sklearn(
        sklearn.pipeline.Pipeline([("standard", StandardScaler())]).fit(X),
        features=X.columns,
    )
    assert pipeline.inference(X).shape == X.shape
    assert pipeline.inference(values).shape == X.shape
    assert pipeline.raw is None
    assert pipeline.features == features

    pipeline = SklearnPipeline.from_sklearn(
        sklearn.pipeline.Pipeline([("standard", StandardScaler())]).fit(X),
        raw=SklearnPipeline.to_raw(X),
    )
    assert pipeline.inference(X).shape == X.shape
    assert pipeline.inference(values).shape == X.shape
    assert pipeline.raw == SklearnPipeline.to_raw(X)
    assert pipeline.features == features

    with pytest.raises(Exception):
        Pipeline.from_sklearn(
            sklearn.pipeline.Pipeline([("standard", StandardScaler())]).fit(X)
        )


def test_sklearn_predict_classification(iris):
    df, features, target = iris
    X, y = df[features], df[target]
    pipeline = SklearnPipeline.from_sklearn(
        sklearn.pipeline.Pipeline([("regression", LogisticRegression())])
    ).fit(X, y)
    assert pipeline.output_columns in pipeline.inference(X)
    assert pipeline.raw == SklearnPipeline.to_raw(X)
    assert pipeline.features == features
    assert pipeline.target == target

    self = pipeline = SklearnPipeline.from_sklearn(
        sklearn.pipeline.Pipeline([("regression", LogisticRegression())]),
        features=features,
        target=target,
    ).fit(df)
    assert pipeline.validate(X, check_na=False)
    pipeline = validate_persistence(pipeline)
    assert pipeline.output_columns in pipeline.inference(X)
    assert pipeline.raw == SklearnPipeline.to_raw(X)
    assert pipeline.features == features
    assert pipeline.target == target


def test_sklrean_predict_regression(iris):
    df, features, target = load_iris()
    X, y = df[features], df[target]
    pipeline = SklearnPipeline.from_sklearn(
        sklearn.pipeline.Pipeline([("regression", LinearRegression())]),
        output_columns=['prediction']
    ).fit(X, y)
    assert pipeline.output_columns[0] in pipeline.inference(X)
    assert pipeline.raw == SklearnPipeline.to_raw(X)
    assert pipeline.features == features
    assert pipeline.target == target

    pipeline = SklearnPipeline.from_sklearn(
        sklearn.pipeline.Pipeline([("regression", LinearRegression())]),
        features=features,
        target=target,
    ).fit(df)
    assert pipeline.output_columns[0] in pipeline.inference(X)
    assert pipeline.raw == SklearnPipeline.to_raw(X)
    assert pipeline.features == features
    assert pipeline.target == target

    predictions = pipeline.predict(X)
    assert len(predictions) == len(X)
    assert isinstance(predictions, (list, np.ndarray))


@pytest.fixture(autouse=True)
def test_skleran_advance(tmpdir, caplog):
    df = pd.read_csv("data/titanic.csv")
    train, test = train_test_split(df)

    target = "Survived"
    features = list(train.columns)
    features.remove(target)

    class PandasTransformer(TransformerMixin, BaseEstimator):
        def fit(self, X, y=None, **fit_params):
            return self

    class FamilySizeTransformer(PandasTransformer):
        def __init__(self, columns):
            self.columns = columns

        def transform(self, df, **transform_params):
            df["FamilySize"] = 1
            for column in self.columns:
                df["FamilySize"] = df["FamilySize"] + df[column]
            return df

    class InitialsTransformer(PandasTransformer):
        def __init__(self, column):
            self.column = column
            self.initials_map = {
                k: v
                for k, v in (
                    zip(
                        [
                            "Miss",
                            "Mr",
                            "Mrs",
                            "Mlle",
                            "Mme",
                            "Ms",
                            "Dr",
                            "Major",
                            "Lady",
                            "Countess",
                            "Jonkheer",
                            "Col",
                            "Rev",
                            "Capt",
                            "Sir",
                            "Don",
                        ],
                        [
                            "Miss",
                            "Mr",
                            "Mrs",
                            "Miss",
                            "Miss",
                            "Miss",
                            "Mr",
                            "Mr",
                            "Mrs",
                            "Mrs",
                            "Other",
                            "Other",
                            "Other",
                            "Mr",
                            "Mr",
                            "Mr",
                        ],
                    )
                )
            }

        def transform(self, df, **transform_params):
            df["Initial"] = df[self.column].str.extract(r"([A-Za-z]+)\.")
            df["Initial"] = df["Initial"].map(self.initials_map)
            return df

    class AgeImputer(PandasTransformer):
        def __init__(self, column):
            self.column = column
            self.means = {}

        def fit(self, X, y=None, **fit_params):
            self.means = (
                X.groupby(["Initial"])["Age"].mean().round().astype(int).to_dict()
            )
            return self

        def transform(self, df, **transform_params):
            for initial, value in self.means.items():
                df["Age"] = np.where(
                    (df["Age"].isnull()) & (df["Initial"].str.match(initial)),
                    value,
                    df["Age"],
                )
            return df

    class AgeGroupTransformer(PandasTransformer):
        def __init__(self, column):
            self.column = column

        def transform(self, df, **transform_params):
            df["AgeGroup"] = None
            df.loc[((df["Sex"] == "male") & (df["Age"] <= 15)), "AgeGroup"] = "boy"
            df.loc[((df["Sex"] == "female") & (df["Age"] <= 15)), "AgeGroup"] = "girl"
            df.loc[
                ((df["Sex"] == "male") & (df["Age"] > 15)), "AgeGroup"
            ] = "adult male"
            df.loc[
                ((df["Sex"] == "female") & (df["Age"] > 15)), "AgeGroup"
            ] = "adult female"
            return df

    class BinTransformer(PandasTransformer):
        def __init__(self, column, bins=None):
            self.column = column
            self.bins = bins or [0, 1, 2, 5, 7, 100, 1000]

        def transform(self, df, **transform_params):
            df["FamilyBin"] = pd.cut(df[self.column], self.bins).astype(str)
            return df

    class MultiColumnLabelEncoder(PandasTransformer):
        def __init__(self, columns=None, prefix="le_", fillna_value=""):
            self.columns = columns
            self.encoders = {}
            self.prefix = prefix
            self.fillna_value = fillna_value

        def _add_prefix(self, col):
            return f"{self.prefix}{col}"

        def preprocess_series(self, s):
            return s.fillna(self.fillna_value).values.reshape(-1, 1)

        def encode(self, column, X):
            return (
                self.encoders[column]
                    .transform(self.preprocess_series(X[column]))
                    .reshape(-1)
            )

        def fit(self, X, y=None):
            for column in self.columns:
                le = OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                )
                self.encoders[column] = le
                le.fit(self.preprocess_series(X[column]))
            return self

        def transform(self, X):
            output = X.copy()
            if self.columns is not None:
                for column in self.columns:
                    output[self._add_prefix(column)] = self.encode(column, X)
            return output

    class LGBMTransformer(PandasTransformer):
        def __init__(self, target, features, output_column="prediction", **params):
            self.features = features
            self.params = params
            self.model = None
            self.target = target
            self.output_column = output_column

        def fit(self, X, y):
            self.model = LGBMClassifier(**self.params).fit(
                X[self.features], X[self.target]
            )
            return self

        def transform(self, df, **transform_params):
            if self.model is None:
                raise RuntimeError("Model is not trained")
            missing_features = [
                feature for feature in self.features if feature not in df
            ]
            if len(missing_features) > 0:
                raise RuntimeError(f"Features missing: {missing_features}")

            df["prediction"] = self.model.predict(df[self.features])
            probabilities = self.model.predict_proba(df[self.features])
            df["probabilities"] = [
                {"died": p[0], "survived": p[1]} for p in probabilities
            ]
            df["label"] = df["prediction"].map({1: "survived", 0: "died"})
            return df

    class CleaningTransformer(PandasTransformer):
        def __init__(self, column):
            self.column = column

        def fit_transform(self, df):
            return df[df[self.column].str.contains(" ") != True]

        def transform(self, df, **transform_params):
            return df

    sk_pipeline = sklearn.pipeline.Pipeline(
        [
            ("cleaning", CleaningTransformer("Cabin")),
            ("FamilySizeTransformer", FamilySizeTransformer(["Parch", "SibSp"])),
            ("InitialsTransformer", InitialsTransformer("Name")),
            ("AgeImputer", AgeImputer("Age")),
            ("AgeGroupTransformer", AgeGroupTransformer("Age")),
            ("BinTransformer", BinTransformer("FamilySize")),
            (
                "MultiColumnLabelEncoder",
                MultiColumnLabelEncoder(columns=["Embarked", "Sex", "FamilyBin"]),
            ),
            (
                "model",
                LGBMTransformer(
                    target="Survived",
                    features=[
                        "PassengerId",
                        "Pclass",
                        "Age",
                        "SibSp",
                        "Parch",
                        "Fare",
                        "le_Embarked",
                        "le_Sex",
                        "le_FamilyBin",
                    ],
                    verbose=-1,
                ),
            ),
        ]
    )

    pipeline = Pipeline.from_sklearn(sk_pipeline).fit(train)

    assert pipeline.inference(test).head(5).shape == (5, 22)
    assert len(pipeline.inference(test)) != len(test)

    with caplog.at_level(logging.WARNING):
        assert "Pipeline doesn't handle NA for PassengerId" in caplog.text
    pipeline.validate(test)

    with pytest.raises(Exception):
        pipeline.validate()

    with pytest.raises(Exception):
        SklearnPipeline.from_sklearn(pipeline.pipeline[1:]).fit(train)

    sk_pipeline = sklearn.pipeline.Pipeline(
        [
            ("cleaning", CleaningTransformer("Cabin")),
            ("FamilySizeTransformer", FamilySizeTransformer(["Parch", "SibSp"])),
            ("InitialsTransformer", InitialsTransformer("Name")),
            ("AgeImputer", AgeImputer("Age")),
            ("AgeGroupTransformer", AgeGroupTransformer("Age")),
            ("BinTransformer", BinTransformer("FamilySize")),
            (
                "MultiColumnLabelEncoder",
                MultiColumnLabelEncoder(columns=["Embarked", "Sex", "FamilyBin"]),
            ),
            (
                "model",
                LGBMTransformer(
                    target="Survived",
                    features=[
                        "PassengerId",
                        "Pclass",
                        "Age",
                        "SibSp",
                        "Parch",
                        "Fare",
                        "le_Embarked",
                        "le_Sex",
                        "le_FamilyBin",
                    ],
                    verbose=-1,
                ),
            ),
        ]
    )

    pipeline = SklearnPipeline.from_sklearn(sk_pipeline[1:]).fit(train)
    pipeline = validate_persistence(pipeline)
    assert pipeline.validate()
    assert pipeline.inference(test).shape == (len(test), 22)

    sample = SklearnPipeline.to_raw(train)
    sample.pop(target)
    assert pipeline.inference(sample).shape == (1, 22)
