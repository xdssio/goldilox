import pandas as pd
from sklearn.datasets import load_iris

# Get teh data
iris = load_iris()
features = iris.feature_names
df = pd.DataFrame(iris.data, columns=features)
df['target'] = iris.target

df.head(2)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import sklearn.pipeline
from goldilox import Pipeline

# Option 1: First create goldilox pipeline, then train (recommended)
sklearn_pipeline = sklearn.pipeline.Pipeline([('standar', StandardScaler()), ('classifier', LinearRegression())])
pipeline = Pipeline.from_sklearn(sklearn_pipeline).fit(df[features], df['target'])

# Options 2: Train model/sklearn-pipeline, then create goldilox pipeline + example of raw
sklearn_pipeline = sklearn_pipeline.fit(df[features], df['target'])
raw = Pipeline.to_raw(df[features])
pipeline = Pipeline.from_sklearn(sklearn_pipeline, raw=raw)

assert pipeline.validate()
