import numpy as np
import pytest
import vaex

from goldilox import Pipeline


@pytest.fixture()
def df():
    # df = vaex.example().head(1000)
    return vaex.example().head(1000)


def test_hnswlib(df):
    import hnswlib

    p = hnswlib.Index(space='l2', dim=df.shape[1] - 1)  # possible options are l2, cosine or ip
    p.init_index(max_elements=len(df), ef_construction=200, M=16)
    features = df.get_column_names(regex='^(?!id|\\.).*')  # not the id
    for i1, i2, chunk in df.to_pandas_df(chunk_size=10000):
        X = chunk[features]
        y = chunk['id']
        p.add_items(X, y)

    p.set_ef(50)  # ef should always be > k (Controlling the recall by setting ef)
    sample = Pipeline._sample_df(df)

    @vaex.register_function(on_expression=False)
    def topk(*columns, k=3):
        labels, _ = p.knn_query(np.array(columns).T, k=k)
        return np.array(labels)

    df['knn'] = df.func.topk(*tuple([df[col] for col in features]), k=3)
    df.add_function('topk', topk)
    pipeline = Pipeline.from_vaex(df)
    assert pipeline.sample == sample
    assert df.to_records(0)['knn'] == [0, 21, 24]
