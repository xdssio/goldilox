def load_iris():
    import pandas as pd
    from sklearn.datasets import load_iris

    iris = load_iris()
    features = iris.feature_names
    features = [feature.replace(' (cm)', '').replace(' ', '_') for feature in features]
    df = pd.DataFrame(iris.data, columns=features)
    df['target'] = iris.target
    return df, features, 'target'
