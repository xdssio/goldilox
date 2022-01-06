import os
import pickle

import numpy as np
import pyarrow as pa
import vaex
from implicit.nearest_neighbours import bm25_weight
from scipy.sparse import csr_matrix

from goldilox import Pipeline
from tests.test_utils import validate_persistence


def test_implicit_als(tmpdir):
    df = vaex.open('data/imdb.parquet')
    userid = 'userId'
    itemid = 'movieId'
    counts = df[itemid].value_counts()
    counts = counts[counts > 100]
    df = df[df[itemid].isin(counts.index)]  # popular movies
    unique_movies = df.groupby([itemid, 'genres']).agg({'title': 'count'})
    # genres = {movie: genres for movie, genres in
    #           zip(unique_movies[itemid].tolist(), unique_movies['genres'].tolist())}
    unique_movies = df.groupby([itemid, 'title']).agg({'count': 'count'})
    titles = {movie: name for movie, name in
              zip(unique_movies[itemid].tolist(), unique_movies['title'].tolist())}

    min_rating = 4.0
    df = df[min_rating < df['rating']]  # liked movies
    ratings = csr_matrix((np.ones(len(df)), (df[itemid].values, df[userid].values)))
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    mean_rating = df['rating'].mean()
    weighted = (bm25_weight(ratings, B=0.9) * 5).tocsr()

    """ Proper training
    from implicit.als import AlternatingLeastSquares
    als = AlternatingLeastSquares(factors=32)
    als.fit(weighted)
    with open('tests/models/als.pkl', 'wb') as handle:
        pickle.dump(als,handle)
    """

    with open('tests/models/als.pkl', 'rb') as handle:
        als = pickle.load(handle)
    """ Proper training
    from implicit.nearest_neighbours import TFIDFRecommender
    tfidf = TFIDFRecommender()
    tfidf.fit(ratings)
    with open('tests/models/tfidf.pkl', 'wb') as handle:
        pickle.dump(tfidf,handle)
    """
    with open('tests/models/tfidf.pkl', 'rb') as handle:
        tfidf = pickle.load(handle)

    """Eye test
    users = df[userid].unique()
    single_user = choice(users)
    user_history = user_items.getrow(single_user).indices
    recommendations = als.recommend(single_user, user_items, N=10, filter_already_liked_items=True)
    """
    user_items = ratings.T.tocsr()

    @vaex.register_function()
    def recommend_als(ar, topk=5, filter_already_liked_items=True):
        ret = []
        for user in ar.tolist():
            recommendations = als.recommend(user, user_items, N=topk,
                                            filter_already_liked_items=filter_already_liked_items)
            recommendation = [titles.get(recommendation[0]) for recommendation in recommendations]
            ret.append(recommendation)
        return pa.array(ret)

    @vaex.register_function()
    def recommend_tfidf(ar, topk=5, filter_already_liked_items=True):
        ret = []
        for user in ar.tolist():
            recommendations = tfidf.recommend(user, user_items, N=topk,
                                              filter_already_liked_items=filter_already_liked_items)
            recommendation = [titles.get(recommendation[0]) for recommendation in recommendations]
            ret.append(recommendation)
        return pa.array(ret)

    @vaex.register_function(on_expression=False)
    def explain(users, items):
        ret = []
        for user, item in zip(users.tolist(), items.tolist()):
            score_explained, contributions, W = als.explain(user, user_items, itemid=item)
            items = [i for i, _ in contributions]
            ret.append([titles.get(i) for i in items])
        return pa.array(ret)

    df.add_function('recommend_als', recommend_als)
    df.add_function('recommend_tfidf', recommend_tfidf)
    df.add_function('explain', explain)

    df['als'] = df[userid].recommend_als()
    df['tfidf'] = df[userid].recommend_tfidf()
    df['explanation'] = df.func.explain(df[userid], df[itemid])

    pipeline = Pipeline.from_vaex(df)
    pipeline = validate_persistence(pipeline)
    assert pipeline.inference({userid: [1, 2, 3]}, columns=[userid, 'als', 'tfidf']).shape == (3, 3)
    assert pipeline.inference({userid: [1, 2, 3], itemid: [2, 2, 2]},
                              columns=[userid, 'als', 'tfidf', 'explanation']).shape == (3, 4)  # TODO
