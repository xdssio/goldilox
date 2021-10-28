from tempfile import NamedTemporaryFile

import numpy as np
import pytest
import traitlets
import vaex
from scipy.sparse import csr_matrix
from sklearn.base import TransformerMixin, BaseEstimator

from goldilox import Pipeline


features = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'E', 'L', 'Lz', 'FeH']


@pytest.fixture()
def df():
    # df = vaex.example().head(1000)
    return vaex.example().head(1000)


def test_hnswlib_vaex(df):
    import hnswlib

    index = hnswlib.Index(space='l2', dim=df.shape[1] - 1)  # possible options are l2, cosine or ip
    index.init_index(max_elements=len(df), ef_construction=200, M=16)
    features = df.get_column_names(regex='^(?!id|\\.).*')  # not the id
    for i1, i2, chunk in df.to_pandas_df(chunk_size=10000):
        X = chunk[features]
        y = chunk['id']
        index.add_items(X, y)

    index.set_ef(50)  # ef should always be > k (Controlling the recall by setting ef)
    sample = Pipeline._sample(df)

    @vaex.register_function(on_expression=False)
    def topk(*columns, k=3):
        labels, _ = index.knn_query(np.array(columns).T, k=k)
        return np.array(labels)

    df['knn'] = df.func.topk(*tuple([df[col] for col in features]), k=3)
    df.add_function('topk', topk)
    pipeline = Pipeline.from_vaex(df)
    assert pipeline.raw == sample
    assert df.to_records(0)['knn'] == [0, 21, 24]


def test_nmslib_vaex(df):
    import nmslib

    ids = {index: _id for index, _id in enumerate(df['id'].tolist())}
    df.variables['id_map'] = ids  # good practice
    X = df[features]

    method = 'hnsw'
    space = 'l2'
    index = nmslib.init(method=method, space=space)
    index.addDataPointBatch(X)
    index.createIndex()

    # we need a pickable model
    class NMSLibModel(traitlets.HasTraits):

        def __init__(self, index=None, method='hnsw', space='cosinesimil'):

            self.method = method
            self.space = space
            self.index = self.decode(index)

        def __reduce__(self):
            return (self.__class__, (self.encode(), self.method, self.space))

        def decode(self, encoding):
            import nmslib
            if isinstance(encoding, bytes):
                index = nmslib.init(method=self.method, space=self.space)
                path = NamedTemporaryFile().name
                with open(path, 'wb') as outfile:
                    outfile.write(encoding)
                index.loadIndex(path)
                return index
            else:
                return encoding

        def encode(self):
            if isinstance(self.index, bytes):
                return self.index
            path = NamedTemporaryFile().name
            self.index.saveIndex(path, save_data=True)
            with open(path, 'rb') as outfile:
                encoding = outfile.read()
            return encoding

        def predict(self, data, k=3):
            neighbours = self.index.knnQueryBatch(data, k=k)
            return np.array(neighbours)[:, 0]

    model = NMSLibModel(index, method, space)

    @vaex.register_function(on_expression=False)
    def topk(*columns, k=3):
        data = np.array(columns).T
        return model.predict(data, k)

    df['knn'] = df.func.topk(*tuple([df[col] for col in features]), k=3)
    df.add_function('topk', topk)

    @vaex.register_function(on_expression=True)
    def results(ar):
        return np.vectorize(ids.get)(ar)

    df.add_function('results', results)
    df['neighbours'] = df['knn'].results()

    pipeline = Pipeline.from_vaex(df)
    assert pipeline.raw
    assert pipeline.inference(pipeline.raw).shape == (1, 13)
    pipeline.save('tests/models/nmslib.pkl')
    from goldilox.vaex.pipeline import VaexPipeline


@pytest.mark.skip("Annoy-Process finished with exit code 132 (interrupted by signal 4: SIGILL)")
def test_annoy_sklearn(df):
    import annoy
    import sklearn.pipeline

    df = vaex.example().head(1000)

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


def test_nmslib_sklearn(df):
    """
    https://scikit-learn.org/stable/auto_examples/neighbors/approximate_nearest_neighbors.html
    """
    import nmslib
    n = 1000
    df = vaex.example().head(n)

    class NMSlibTransformer(TransformerMixin, BaseEstimator):
        """Wrapper for using nmslib as sklearn's KNeighborsTransformer"""

        def __init__(self, n_neighbors=5, output_column='knn', metric="euclidean", method="sw-graph", n_jobs=1):
            self.n_neighbors = n_neighbors
            self.method = method
            self.metric = metric
            self.n_jobs = n_jobs
            self.output_column = output_column
            self.n_samples_fit_ = None

        def __sklearn_is_fitted__(self):
            return self.n_samples_fit_ is not None

        def fit(self, X, y=None):
            self.n_samples_fit_ = X.shape[0]

            # see more metric in the manual
            # https://github.com/nmslib/nmslib/tree/master/manual
            space = {
                "euclidean": "l2",
                "cosine": "cosinesimil",
                "l1": "l1",
                "l2": "l2",
            }[self.metric]

            self.nmslib_ = nmslib.init(method=self.method, space=space)
            self.nmslib_.addDataPointBatch(X)
            self.nmslib_.createIndex()
            return self

        def transform(self, X):
            results = self.nmslib_.knnQueryBatch(X, k=self.n_neighbors, num_threads=self.n_jobs)
            indices, distances = zip(*results)
            indices, distances = np.vstack(indices), np.vstack(distances)
            X[self.output_column] = tuple(indices)
            return X

    X = df[features].to_pandas_df()

    sample = X.to_records(0)
    pipeline = Pipeline.from_sklearn(NMSlibTransformer()).fit(X)

    assert pipeline.inference(sample).shape == (n, 11)


