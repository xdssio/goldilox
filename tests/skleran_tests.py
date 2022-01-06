import warnings

import numpy as np
import pandas as pd
import pytest
import sklearn.pipeline
from lightgbm.sklearn import LGBMClassifier
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

from tests.test_utils import validate_persistence

warnings.filterwarnings("ignore")

from goldilox.sklearn.pipeline import SklearnPipeline
from vaex.ml.datasets import load_iris


@pytest.fixture()
def iris():
    # iris = load_iris().to_pandas_df()
    return load_iris().to_pandas_df()


def test_from_sklearn_transform(iris):
    columns = ["petal_length", "petal_width", "sepal_length", "sepal_width"]
    X = iris[columns]
    values = X.values
    pipeline = SklearnPipeline.from_sklearn(
        sklearn.pipeline.Pipeline([("standard", StandardScaler())])
    ).fit(values)
    assert pipeline.inference(X).shape == X.shape
    assert pipeline.inference(values).shape == X.shape
    assert pipeline.raw == list(values[0])

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
    assert pipeline.features == columns

    pipeline = SklearnPipeline.from_sklearn(
        sklearn.pipeline.Pipeline([("standard", StandardScaler())]).fit(X),
        raw=SklearnPipeline.to_raw(X),
    )
    assert pipeline.inference(X).shape == X.shape
    assert pipeline.inference(values).shape == X.shape
    assert pipeline.raw == SklearnPipeline.to_raw(X)
    assert pipeline.features == columns

    with pytest.raises(Exception):
        SklearnPipeline.from_sklearn(
            sklearn.pipeline.Pipeline([("standard", StandardScaler())]).fit(X)
        )


def test_sklrean_predict_classification(iris):
    df = iris.to_pandas_df()
    features = ["petal_length", "petal_width", "sepal_length", "sepal_width"]
    target = "class_"
    X = df[features]
    y = df[target]
    pipeline = SklearnPipeline.from_sklearn(
        sklearn.pipeline.Pipeline([("regression", LogisticRegression())])
    ).fit(X, y)
    assert pipeline.output_column in pipeline.inference(X)
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
    assert pipeline.output_column in pipeline.inference(X)
    assert pipeline.raw == SklearnPipeline.to_raw(X)
    assert pipeline.features == features
    assert pipeline.target == target


def test_sklrean_predict_regression(iris):
    features = ["petal_length", "petal_width", "sepal_length", "sepal_width"]
    target = "class_"
    X = iris[features]
    y = iris[target]
    pipeline = SklearnPipeline.from_sklearn(
        sklearn.pipeline.Pipeline([("regression", LinearRegression())])
    ).fit(X, y)
    assert pipeline.output_column in pipeline.inference(X)
    assert pipeline.raw == SklearnPipeline.to_raw(X)
    assert pipeline.features == features
    assert pipeline.target == target

    pipeline = SklearnPipeline.from_sklearn(
        sklearn.pipeline.Pipeline([("regression", LinearRegression())]),
        features=features,
        target=target,
    ).fit(iris)
    assert pipeline.output_column in pipeline.inference(X)
    assert pipeline.raw == SklearnPipeline.to_raw(X)
    assert pipeline.features == features
    assert pipeline.target == target


def test_skleran_advance(tmpdir):
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

        def transform(self, df, **transform_params):
            return df[df[self.column].str.contains(" ") != True]

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

    pipeline = SklearnPipeline.from_sklearn(sk_pipeline).fit(train)

    assert pipeline.inference(test).head(5).shape == (5, 22)
    assert len(pipeline.inference(test)) != len(test)

    with pytest.raises(Exception):
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
