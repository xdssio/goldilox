import pandas as pd
import sklearn.datasets


def load_iris():
    iris = sklearn.datasets.load_iris()
    features = iris.feature_names
    features = [feature.replace(' (cm)', '').replace(' ', '_') for feature in features]
    df = pd.DataFrame(iris.data, columns=features)
    df['target'] = iris.target
    return df, features, 'target'


def make_blobs(n_features: int = 10):
    features = [f"feature{i}" for i in range(n_features)]
    blobs, labels = sklearn.datasets.make_blobs(n_samples=2000, n_features=n_features)
    df = pd.DataFrame(blobs, columns=features)
    df['target'] = labels
    return df, features, 'target'
