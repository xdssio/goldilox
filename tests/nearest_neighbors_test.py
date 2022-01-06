from tempfile import NamedTemporaryFile

import numpy as np
import pytest
import traitlets
import vaex
from scipy.sparse import csr_matrix
from sklearn.base import TransformerMixin, BaseEstimator

from goldilox import Pipeline
from tests.test_utils import validate_persistence

features = ["x", "y", "z", "vx", "vy", "vz", "E", "L", "Lz", "FeH"]


def test_hnswlib_vaex():
    import hnswlib

    df = vaex.example().head(1000)

    index = hnswlib.Index(
        space="l2", dim=df.shape[1] - 1
    )  # possible options are l2, cosine or ip
    index.init_index(max_elements=len(df), ef_construction=200, M=16)
    features = df.get_column_names(regex="^(?!id|\\.).*")  # not the id
    for i1, i2, chunk in df.to_pandas_df(chunk_size=10000):
        X = chunk[features]
        y = chunk["id"]
        index.add_items(X, y)

    index.set_ef(50)  # ef should always be > k (Controlling the recall by setting ef)
    sample = Pipeline.to_raw(df)

    @vaex.register_function(on_expression=False)
    def topk(*columns, k=3):
        labels, _ = index.knn_query(np.array(columns).T, k=k)
        return np.array(labels)

    df["knn"] = df.func.topk(*tuple([df[col] for col in features]), k=3)
    df.add_function("topk", topk)
    pipeline = Pipeline.from_vaex(df)
    pipeline = validate_persistence(pipeline)
    assert pipeline.raw == sample
    assert df.to_records(0)["knn"] == [0, 21, 24]


def test_nmslib_vaex():
    import nmslib

    df = vaex.example().head(1000)
    ids = {index: _id for index, _id in enumerate(df["id"].tolist())}
    df.variables["id_map"] = ids  # good practice
    X = df[features]

    method = "hnsw"
    space = "l2"
    index = nmslib.init(method=method, space=space)
    index.addDataPointBatch(X)
    index.createIndex()

    # we need a pickable model
    class NMSLibModel(traitlets.HasTraits):
        def __init__(self, index=None, method="hnsw", metric="cosinesimil"):

            self.method = method
            self.metric = metric
            self.index = self._decode(index)

        def __reduce__(self):
            return (self.__class__, (self._encode(), self.method, self.metric))

        def _decode(self, encoding):
            if isinstance(encoding, bytes):
                index = nmslib.init(method=self.method, space=self.metric)
                path = NamedTemporaryFile().name
                with open(path, "wb") as outfile:
                    outfile.write(encoding)
                index.loadIndex(path)
                return index
            else:
                return encoding

        def _encode(self):
            if isinstance(self.index, bytes):
                return self.index
            path = NamedTemporaryFile().name
            self.index.saveIndex(path, save_data=True)
            with open(path, "rb") as outfile:
                encoding = outfile.read()
            return encoding

        def predict(self, data, k=3):
            neighbours = self.index.knnQueryBatch(data, k=k)
            return np.array(neighbours)[:, 0]

    model = NMSLibModel(index, method, space)

    @vaex.register_function(on_expression=False)
    def topk(*columns):
        data = np.array(columns).T
        return model.predict(data, 3)

    df.add_function("topk", topk)
    df["knn"] = df.func.topk(*features)

    @vaex.register_function(on_expression=True)
    def results(ar):
        return np.vectorize(ids.get)(ar)

    df.add_function("results", results)
    df["neighbours"] = df["knn"].results()

    pipeline = Pipeline.from_vaex(df)
    pipeline = validate_persistence(pipeline)
    assert pipeline.raw
    assert pipeline.inference(pipeline.raw).shape == (1, 13)


@pytest.mark.skip(
    "Annoy-Process finished with exit code 132 (interrupted by signal 4: SIGILL)"
)
def test_annoy_sklearn(df):
    import annoy
    import sklearn.pipeline

    # df = vaex.example().head(1000)

    class AnnoyTransformer(TransformerMixin, BaseEstimator):
        """Wrapper for using annoy.AnnoyIndex as sklearn's KNeighborsTransformer"""

        def __init__(self, n_neighbors=5, metric="manhattan", n_trees=10, search_k=-1):
            self.n_neighbors = n_neighbors
            self.n_trees = n_trees
            self.search_k = search_k
            self.metric = metric

        def fit(self, X):
            self.n_samples_fit_ = X.shape[0]
            self.annoy_ = annoy.AnnoyIndex(X.shape[1], metric=self.metric)
            for i, x in enumerate(X):
                self.annoy_.add_item(i, x.tolist())
            self.annoy_.build(self.n_trees)
            return self

        def transform(self, X):
            return self._transform(X)

        def fit_transform(self, X, y=None):
            return self.fit(X)._transform(X=None)

        def _transform(self, X):
            """As `transform`, but handles X is None for faster `fit_transform`."""

            n_samples_transform = self.n_samples_fit_ if X is None else X.shape[0]

            # For compatibility reasons, as each sample is considered as its own
            # neighbor, one extra neighbor will be computed.
            n_neighbors = self.n_neighbors + 1

            indices = np.empty((n_samples_transform, n_neighbors), dtype=int)
            distances = np.empty((n_samples_transform, n_neighbors))

            if X is None:
                for i in range(self.annoy_.get_n_items()):
                    ind, dist = self.annoy_.get_nns_by_item(
                        i, n_neighbors, self.search_k, include_distances=True
                    )

                    indices[i], distances[i] = ind, dist
            else:
                for i, x in enumerate(X):
                    indices[i], distances[i] = self.annoy_.get_nns_by_vector(
                        x.tolist(), n_neighbors, self.search_k, include_distances=True
                    )

            indptr = np.arange(0, n_samples_transform * n_neighbors + 1, n_neighbors)
            kneighbors_graph = csr_matrix(
                (distances.ravel(), indices.ravel(), indptr),
                shape=(n_samples_transform, self.n_samples_fit_),
            )

            return kneighbors_graph

    nn = AnnoyTransformer()
    nn.fit(df[features].values)
    sk_pipeline = sklearn.pipeline.Pipeline([("annoy", AnnoyTransformer())])
    sk_pipeline.fit(df)


def test_nmslib_sklearn():
    """
    https://scikit-learn.org/stable/auto_examples/neighbors/approximate_nearest_neighbors.html
    """
    n = 1000
    df = vaex.example().head(n)

    # https://github.com/nmslib/nmslib/tree/master/manual
    class NMSlibTransformer(TransformerMixin, BaseEstimator):
        """Wrapper for using nmslib as sklearn's KNeighborsTransformer"""

        def __init__(
                self,
                n_neighbors=5,
                output_column="knn",
                method="hnsw",
                metric="cosinesimil",
                n_jobs=1,
                index=None,
        ):

            self.n_neighbors = n_neighbors
            self.method = method
            self.metric = metric
            self.n_jobs = n_jobs
            self.output_column = output_column
            self.n_samples_fit_ = None
            self.index = self._create_index(index)

        def __reduce__(self):
            return (
                self.__class__,
                (
                    self.n_neighbors,
                    self.output_column,
                    self.method,
                    self.metric,
                    self.n_jobs,
                    self._encode(),
                ),
            )

        def _create_index(self, encoding):
            import nmslib

            if encoding is None:
                return nmslib.init(method=self.method, space=self.metric)
            if isinstance(encoding, bytes):
                index = nmslib.init(method=self.method, space=self.metric)
                path = NamedTemporaryFile().name
                with open(path, "wb") as outfile:
                    outfile.write(encoding)
                index.loadIndex(path)
                return index
            else:
                return encoding

        def _encode(self):
            if self.index is None:
                return None
            if isinstance(self.index, bytes):
                return self.index
            path = NamedTemporaryFile().name
            self.index.saveIndex(path, save_data=True)
            with open(path, "rb") as outfile:
                encoding = outfile.read()
            return encoding

        def __sklearn_is_fitted__(self):
            return self.n_samples_fit_ is not None

        def fit(self, X, y=None):
            self.n_samples_fit_ = X.shape[0]

            self.index.addDataPointBatch(X)
            self.index.createIndex()
            return self

        def transform(self, X):
            results = self.index.knnQueryBatch(
                X, k=self.n_neighbors, num_threads=self.n_jobs
            )
            indices, distances = zip(*results)
            indices, distances = np.vstack(indices), np.vstack(distances)
            X[self.output_column] = tuple(indices)
            return X

    X = df[features].to_pandas_df()

    sample = X.to_records(0)
    pipeline = Pipeline.from_sklearn(NMSlibTransformer()).fit(X)
    pipeline = validate_persistence(pipeline)
    assert pipeline.inference(sample).shape == (n, 11)


def test_kdtree():
    """
    https://scikit-learn.org/stable/auto_examples/neighbors/approximate_nearest_neighbors.html
    """
    n = 1000
    df = vaex.example().head(n)

    # Vaex version
    from sklearn.neighbors import KDTree

    model = KDTree(df[features], leaf_size=2)

    @vaex.register_function(on_expression=False)
    def query(*columns):
        data = np.array(columns).T
        dist, ind = model.query(data, k=3)
        return ind

    df.add_function("query", query)
    df["predictions"] = df.func.query(*tuple([df[col] for col in features]))
    pipeline = Pipeline.from_vaex(df)

    pipeline = validate_persistence(pipeline)
    assert pipeline.inference(pipeline.raw).shape == (1, 12)

    # SKlearn version
    X = df[features].to_pandas_df()
    model = KDTree(X[features], leaf_size=2)

    class KDTreePredictor(TransformerMixin, BaseEstimator):
        def __init__(self, features=None, leaf_size=2, k=3, output_column="results"):
            self.index = None
            self.ids = None
            self.features = features
            self.k = k
            self.leaf_size = leaf_size
            self.output_column = output_column

        def fit(self, X, y=None):
            if y is not None:
                assert len(X) == len(y)
                self.ids = {i: j for i, j in enumerate(y)}
            if self.features and isinstance(self.features, list):
                X = X[self.features]
            self.index = KDTree(X, leaf_size=self.leaf_size)
            return self

        def transform(self, X):
            copy = X.copy()
            if self.index is None:
                raise RuntimeError("model was not trained")
            if self.features and isinstance(self.features, list):
                copy = X[self.features]
            _, ind = self.index.query(copy, k=self.k)
            copy[self.output_column] = list(ind)
            return copy

    pipeline = Pipeline.from_sklearn(KDTreePredictor()).fit(X)
    pipeline = validate_persistence(pipeline)
    assert pipeline.validate()
    assert pipeline.inference(pipeline.raw).shape == (1, 11)
