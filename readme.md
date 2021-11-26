# What is Goldilox?

Goldilox is a one line tool which transform a machine learning solution into an object for production.   
This is in current development, please wait for the first stable version. 

# Installing
With pip:
```
$ pip install goldilox
```

[For more details, see the documentation](www.todo) #TODO

# Key features
* One line from POC to production
* Flexible and yet simple
* Technology agnostic
* Things you didn't know you want:   
  * Serialization validation
  * Missing values validation
  * Output validation
  * I/O examples
  * Variables and description queries

# Vaex First
[Vaex]((https://github.com/vaexio/vaex)) is an open-soruce big data technology with similar APIs to [Pandas](https://pandas.pydata.org/).   
We use some of the Vaex special sauce to allow the extreme flexibility for advance pipeline solutions while insuring we have a tool that works on big data.
* [![Documentation](https://readthedocs.org/projects/vaex/badge/?version=latest)](https://docs.vaex.io)

# Pandas + Sklearn support
Any [Sklearn](https://scikit-learn.org/) + [Pandas](https://pandas.pydata.org/) pipeline/transformer/estimator works as well.



# Examples   

**1. Data science**    
Vaex
```python
import vaex
from vaex.ml.datasets import load_iris
from vaex.ml.xgboost import XGBoostModel
import numpy as np

df = load_iris()

# feature engineering example
df["petal_ratio"] = df["petal_length"] / df["petal_width"]

# modeling
booster = XGBoostModel(
    features=['petal_length', 'petal_width', 'sepal_length', 'sepal_width', 'petal_ratio'],
    target='class_',
    prediction_name="prediction",
    num_boost_round=500,
)
booster.fit(df)
df = booster.transform(df)

# post modeling procssing example 
df['prediction'] = np.around(df['prediction'])
df["label"] = df["prediction"].map({0: "setosa", 1: "versicolor", 2: "virginica"})
```
SKlearn
```python
import pandas as pd
import json
from xgboost.sklearn import XGBClassifier
from sklearn.datasets import load_iris

# Get teh data
iris  = load_iris()
features = iris.feature_names
df = pd.DataFrame(iris.data, columns=features)
df['target'] = iris.target


model  = LGBMClassifier().fit(df[features], df['target'])
```
**2. Build a production ready pipeline**   
* In one line (-:
```python
from goldilox import Pipeline

# vaex
pipeline = Pipeline.from_vaex(df) 

# sklearn - When using sklearn, we want to have an example of the raw production query data
pipeline = Pipeline.from_sklearn(model, raw=Pipeline.to_raw(df[features])) 

# Validate
assert pipeline.validate()

# Save and load
pipeline.save(<path>)
pipeline = Pipeline.from_file(<path>)
```
**3. Deploy**
```
gl serve <path>

[2021-11-16 18:54:44 +0100] [74906] [INFO] Starting gunicorn 20.1.0
[2021-11-16 18:54:44 +0100] [74906] [INFO] Listening at: http://127.0.0.1:5000 (74906)
[2021-11-16 18:54:44 +0100] [74906] [INFO] Using worker: uvicorn.workers.UvicornH11Worker
[2021-11-16 18:54:44 +0100] [74911] [INFO] Booting worker with pid: 74911
[2021-11-16 18:54:44 +0100] [74911] [INFO] Started server process [74911]
[2021-11-16 18:54:44 +0100] [74911] [INFO] Waiting for application startup.
[2021-11-16 18:54:44 +0100] [74911] [INFO] Application startup complete.
```
**4. Training:**  For experiments, cloud training, automations, etc,.    

With *Vaex*, you put everything you want to do to a function which recives and returns a Vaex DataFrame

```python
from vaex.ml.datasets import load_iris
from goldilox import Pipeline

def fit(df):  
  from vaex.ml.xgboost import XGBoostModel
  import numpy as np

  df = load_iris()

  # feature engineering example
  df["petal_ratio"] = df["petal_length"] / df["petal_width"]

  # modeling
  booster = XGBoostModel(
      features=['petal_length', 'petal_width', 'sepal_length', 'sepal_width', 'petal_ratio'],
      target='class_',
      prediction_name="prediction",
      num_boost_round=500,
  )
  booster.fit(df)
  df = booster.transform(df)

  # post modeling procssing example 
  df['prediction'] = np.around(df['prediction'])
  df["label"] = df["prediction"].map({0: "setosa", 1: "versicolor", 2: "virginica"})
  return df


df = load_iris()
pipeline = Pipeline.from_vaex(df,fit=fit).fit(df)
```
With *Sklearn* the fit would be the standard X and y.
```python
import pandas as pd
import sklearn.pipeline
from sklearn.datasets import load_iris
from xgboost.sklearn import XGBClassifier
from sklearn.pipeline import Pipeline as SklearnPipeline

iris  = load_iris()
features = iris.feature_names
df = pd.DataFrame(iris.data, columns=features)
df['target'] = iris.target

# we don't need to provide raw example if we do the training - it would be sampeld automatically.
classifier = XGBClassifier(n_estimators=10, verbosity=0, use_label_encoder=False)
pipeline = Pipeline.from_sklearn(classifier).fit(df[features], df['target'])
assert pipeline.validate()

>>> Pipeline doesn't handle na for sepal length (cm)
>>> Pipeline doesn't handle na for sepal width (cm)
>>> Pipeline doesn't handle na for petal length (cm)
>>> Pipeline doesn't handle na for petal width (cm)
```
We do not handle missing values? Let's fix that!

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

imputer = ColumnTransformer([('features_mean', SimpleImputer(strategy='mean'), \
                              features)], remainder='passthrough')                              
classifier = XGBClassifier(n_estimators=10, verbosity=0, use_label_encoder=False)

sk_pipeline = sklearn.pipeline.Pipeline([('imputer', imputer), ('classifier', classifier)]) 
pipeline = Pipeline.from_sklearn(sk_pipeline).fit(df[features], df['target'])
assert pipeline.validate()                              
```
* We can still deploy a pipeline that doesn't deal with missing values if we want, *validate()* returns `True` if serialization, and prediction of raw validations pass.

# [Notebook examples](https://github.com/xdssio/goldilox/tree/master/notebooks)
* **Classification/Regression** 
  * [LightGBM notebook](https://github.com/xdssio/goldilox/blob/master/notebooks/lightgbm.ipynb)
  * [XGBoost notebook](https://github.com/xdssio/goldilox/blob/master/notebooks/xgboost.ipynb) 
  * [Catbboost](https://github.com/xdssio/goldilox/blob/master/notebooks/catboost.ipynb)
  * [Skleran](https://github.com/xdssio/goldilox/blob/master/notebooks/skleran_simple.ipynb)
  
* **Clustering**
  * [Kmeans](https://github.com/xdssio/goldilox/blob/master/notebooks/clustering.ipynb)
    
* **Nearest neighbours**
  * [KTTree (sklearn), hnswlib, nmslib](https://github.com/xdssio/goldilox/blob/master/notebooks/nearest_neighbors.ipynb)
  
* **Recommendations**
  * [Implicit (Matrix Factorization)](https://github.com/xdssio/goldilox/blob/master/notebooks/implicit.ipynb)

* **NLP**
  * [TFIDF (Sklearn)]() #TODO
  * [Spacy]() #TODO
  * [Transformers]() #TODO
    
* **Deep learning**
  * [Keras]() #TODO
  * [Tensorflow]() #TODO
  * [PyTorch]() #TODO
  * [MXNet]() #TODO
    
* Online learning
 * [River]() #TODO

* Explnations
  * [SHAP]() #TODO

* Training
  * [AIM](https://github.com/aimhubio/aim) #TODO

* **Advance pipelines**
  * [LightGBM with Vaex]() #TODO
  * [Sklearn]() # TODO

# [Roadmap](https://github.com/xdssio/goldilox/wiki/Roadmap)