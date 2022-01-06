import pytest
from vaex.ml.datasets import load_iris

from goldilox import Pipeline


def lightgbm_vaex_fit():
    def fit(df):
        import vaex
        import numpy as np
        from vaex.ml.lightgbm import LightGBMModel
        from sklearn.metrics import accuracy_score

        train, test = df.ml.train_test_split(test_size=0.2, verbose=False)

        features = ["petal_length", "petal_width", "sepal_length", "sepal_width"]
        target = "class_"

        booster = LightGBMModel(
            features=features,
            target=target,
            prediction_name="predictions",
            num_boost_round=500,
            params={"verbose": -1, "objective": "multiclass", "num_class": 3},
        )
        booster.fit(df)

        @vaex.register_function()
        def argmax(ar, axis=1):
            return np.argmax(ar, axis=axis)

        train = booster.transform(df)
        train.add_function("argmax", argmax)
        train["prediction"] = train["predictions"].argmax()

        pipeline = Pipeline.from_vaex(train)
        accuracy = accuracy_score(
            pipeline.inference(test[features])["prediction"].values, test[target].values
        )
        booster = LightGBMModel(
            features=features,
            target=target,
            prediction_name="predictions",
            num_boost_round=500,
            params={"verbose": -1, "objective": "multiclass", "num_class": 3},
        )
        booster.fit(df)
        df = booster.transform(df)
        df.add_function("argmax", argmax)
        df["prediction"] = df["predictions"].argmax()
        names = {0: "setosa", 1: "versicolor", 2: "virginica"}
        df["label"] = df["prediction"].map(names)
        df["probabilities"] = df["predictions"].apply(
            lambda x: {names.get(i): x[i] for i in range(3)}
        )
        df.variables["accuracy"] = accuracy
        df.variables["names"] = names
        return df

    iris = load_iris()
    df = iris.copy()
    pipeline = Pipeline.from_vaex(
        df, fit=fit, description="Lightgbm with Vaex"
    )
    data = df.to_records(0)
    assert pipeline.inference(data).shape == df.head(1).shape
    pipeline.fit(df)

    assert pipeline.inference(data).shape == (1, 9)
    assert pipeline.get_variable("accuracy")
    assert pipeline.get_variable("names")
    assert pipeline.raw == data
    assert list(pipeline.example.keys()) == [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "class_",
        "predictions",
        "prediction",
        "label",
        "probabilities",
    ]
    assert "Lightgbm" in pipeline.description
    pipeline.raw.pop("class_")
    assert pipeline.inference(data).shape == (1, 9)
    pipeline.validate(df.head(2))
    print(pipeline._get_packages())

    pipeline.save("../goldilox-ops/models/pipeline.pkl")
    pipeline.save("./pipeline.pkl")


def test_lightgbm_sklearn():
    from lightgbm.sklearn import LGBMClassifier
    import sklearn.pipeline

    df = load_iris().copy()
    features = ["petal_length", "petal_width", "sepal_length", "sepal_width"]
    target = "class_"
    sk_pipeline = sklearn.pipeline.Pipeline([("classifier", LGBMClassifier())])
    X = df[features]
    y = df[target]
    self = pipeline = Pipeline.from_sklearn(
        sk_pipeline, description="Lightgbm with sklearn"
    ).fit(X, y)

    assert pipeline.inference(X).head(10).shape == (10, 5)
    assert pipeline.inference(X.values[:10]).shape == (10, 5)
    assert pipeline.inference(self.raw).shape == (1, 5)

    pipeline.fit(df)
    assert pipeline.inference(X).head(10).shape == (10, 5)
    assert pipeline.inference(X.values[:10]).shape == (10, 5)
    assert pipeline.inference(self.raw).shape == (1, 5)

    # with a trained sklearn pipeline
    sample = X.head(1).to_records()[0]
    self = pipeline = Pipeline.from_sklearn(
        sk_pipeline, raw=sample, description="Lightgbm with sklearn"
    ).fit(X, y)
    assert pipeline.inference(X).head(10).shape == (10, 5)
    assert pipeline.inference(X.values[:10]).shape == (10, 5)
    assert pipeline.inference(self.raw).shape == (1, 5)
    assert "Lightgbm" in pipeline.description
    pipeline.save("../goldilox-ops/models/sk.pkl")


@pytest.mark.skip("test manually")
def test_faiss():
    import vaex
    from faiss import IndexFlatL2
    from goldilox import Pipeline
    import numpy as np
    import traitlets
    from tempfile import NamedTemporaryFile
    from faiss import write_index, read_index

    df = vaex.example().head(1000)
    features = df.get_column_names(regex="[^id]")  # not the ida
    d = len(features)
    X = np.float32(np.ascontiguousarray(df[features]))
    index = IndexFlatL2(d)
    index.add(X)

    class FiassModel(traitlets.HasTraits):

        # This should work with the reduce's arguments
        def __init__(self, index=None):
            self.index = self._decode(index)

        # This is how you make a class pickalbe
        def __reduce__(self):
            return (self.__class__, (self._encode(),))

        # how nmslib implemented serialization
        def _decode(self, encoding):
            if isinstance(encoding, bytes):
                path = NamedTemporaryFile().name
                with open(path, "wb") as outfile:
                    outfile.write(encoding)
                return read_index(path)
            else:
                return encoding

        # how nmslib implemented serialization
        def _encode(self):
            if isinstance(self.index, bytes):
                return self.index
            path = NamedTemporaryFile().name
            write_index(self.index, path)
            with open(path, "rb") as outfile:
                encoding = outfile.read()
            return encoding

        def predict(self, data, k=3):
            data = np.float32(np.ascontiguousarray(data))
            _, ind = model.index.search(data, k)
            return ind

    model = FiassModel(index)

    @vaex.register_function(on_expression=False)
    def search(*columns):
        k = 3
        data = np.float32(np.ascontiguousarray(np.array(columns).T))
        _, ind = model.index.search(data, k)
        return ind

    df.add_function("search", search)
    df["neighbors"] = df.func.search(*features)
    pipeline = Pipeline.from_vaex(df)
    pipeline.validate()
    pipeline.inference(pipeline.raw)
    pipeline.save("../goldilox-ops/models/faiss.pkl")
