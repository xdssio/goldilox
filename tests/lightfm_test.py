from collections import defaultdict

import numpy as np
import pyarrow as pa
import vaex
from lightfm import LightFM
from scipy.sparse import csr_matrix
from vaex.ml import LabelEncoder

from goldilox import Pipeline


def fetch_train():
    from lightfm.datasets import fetch_movielens
    data = fetch_movielens(min_rating=5.0)
    names = {key: value for key, value in enumerate(data['item_labels'])}

    def to_vaex(s):
        dok = s.todok()
        users = []
        movies = []
        ratings = []
        for key, value in dok.items():
            users.append(key[0])
            movies.append(names.get(key[1]))
            ratings.append(value)

        df = vaex.from_arrays(users=users, movies=movies, ratings=ratings)
        return df

    return to_vaex(data['train']), to_vaex(data['test'])


# TODO clean and explain
def test_lightfm():
    # Load the MovieLens 100k dataset. Only five
    # star ratings are treated as positive.
    train = vaex.open('data/imdb100k_train.parquet')
    test = vaex.open('data/imdb100k_test.parquet')

    le = LabelEncoder(features=['movies'])
    train = le.fit_transform(train)
    train['label_encoded_movies'] = train['label_encoded_movies'] + 1
    X = csr_matrix((train['ratings'].values, (train['users'].values, train['label_encoded_movies'].values)))

    model = LightFM(loss='warp')
    model.fit(X, epochs=30, num_threads=4)

    # test_precision = precision_at_k(model, test, k=5).mean()
    # assert 0 < test_precision

    users_history = defaultdict(list)
    all_movies = set(train['label_encoded_movies'].unique())
    for user, movie in zip(train['users'].tolist(), train['label_encoded_movies'].tolist()):
        users_history[user].append(movie)
    users_options = {user: all_movies.difference(history) for user, history in users_history.items()}
    most_popular = list(train['label_encoded_movies'].value_counts()[:5])
    titles = {value: key for key, value in le.labels_['movies'].items()}

    @vaex.register_function()
    def recommend(ar, topk=5):
        ret = []
        for user in ar.tolist():
            user_options = list(users_options.get(user))
            if not user_options:
                ret.append(most_popular)
            else:
                recommendations = model.predict(np.repeat(user, len(user_options)), user_options).argsort()[-topk:][
                                  ::-1]
                if len(recommendations) == 0:
                    recommendations = most_popular
                ret.append(recommendations)
        return np.array(ret)

    @vaex.register_function()
    def topk(ar, k=0):
        ar = [titles.get(i[k], most_popular[k]) if isinstance(i, list) else titles.get(most_popular[k]) for i in ar]
        return pa.array(ar)

    train.add_function('recommend', recommend)
    train['users'].recommend()

    train.add_function('topk', topk)
    train['recommendations_ids'] = train['users'].recommend()
    train['recommendation0'] = train['recommendations_ids'].topk()
    train['recommendation1'] = train['recommendations_ids'].topk(1)

    pipeline = Pipeline.from_vaex(train)

    users = test.head(3)['users'].tolist()
    assert pipeline.inference({'users': np.array(users)},
                              columns=['users', 'recommendations_ids', 'recommendation0', 'recommendation1']).shape[
               1] == 4
    assert pipeline.validate()
