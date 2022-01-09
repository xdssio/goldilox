import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from goldilox import Pipeline

iris = load_iris()
features = iris.feature_names
df = pd.DataFrame(iris.data, columns=features)
df['target'] = iris.target
train, test = train_test_split(df)

pipeline = Pipeline.from_sklearn(LogisticRegression()).fit(train[features], train['target'])
pipeline.inference(test).head(2)

from sklearn.decomposition import PCA

pipeline = Pipeline.from_sklearn(PCA(n_components=2),
                                 output_columns=['pca1', 'pca2']
                                 ).fit(train[features])
pipeline.inference(test).head(2)
