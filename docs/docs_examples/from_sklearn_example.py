import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
features = iris.feature_names
df = pd.DataFrame(iris.data, columns=features)
df['target'] = iris.target
df.head(2)

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SklearnPipline
from goldilox import Pipeline

pipeline = Pipeline.from_sklearn(SklearnPipline([("logistic", LogisticRegression())]),
                                 target='target',
                                 features=features,
                                 output_columns="predictions")

from sklearn.linear_model import LogisticRegression
from goldilox import Pipeline

pipeline = Pipeline.from_sklearn(LogisticRegression()).fit(X, y)
print(f"features: {pipeline.features} \ntarget: {pipeline.target}")
