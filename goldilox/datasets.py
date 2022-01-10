def load_iris(dataframe='pandas'):
    import pandas as pd
    from sklearn.datasets import load_iris

    iris = load_iris()
    features = iris.feature_names
    df = pd.DataFrame(iris.data, columns=features).rename(
        columns={f: f.replace(' (cm)', '').replace(' ', '_') for f in features})
    df['target'] = iris.target
    if dataframe == 'vaex':
        import vaex
        return vaex.from_pandas(df)
    return df
