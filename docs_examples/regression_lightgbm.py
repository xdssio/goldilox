from lightgbm.sklearn import LGBMRegressor
from vaex.ml.datasets import load_iris
from vaex.ml.sklearn import Predictor

df = load_iris()
target = 'class_'

# feature engineering example
df['petal_ratio'] = df['petal_length'] / df['petal_width']
features = ['petal_length', 'petal_width', 'sepal_length',
            'sepal_width', 'petal_ratio']
model = Predictor(model=LGBMRegressor(**{'verbosity': -1,
                                         'objective': 'regression'}),
                  features=features,
                  target=target,
                  prediction_name='prediction')
model.fit(df)
df = model.transform(df)
df.head(2)

from goldilox import Pipeline

pipeline = Pipeline.from_vaex(df)
#
# print(f"Saved to: {pipeline.save('pipeline.pkl')}")
# print(f"Check out the docs: http://127.0.0.1:5000/docs\n")
# # !glx serve pipeline.pkl
