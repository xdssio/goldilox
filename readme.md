# What is Goldilox?

Goldilox is a one line tool which transform a machine learning solution into an object for production.   
This is in current development, please wait for the first stable version.

# Installing

With pip:

```
$ pip install goldilox
```

[For more details, see the documentation](www.docs.goldilox.io)

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

[Vaex](https://github.com/vaexio/vaex) is an open-soruce big data technology with similar APIs
to [Pandas](https://pandas.pydata.org/).   
We use some of the Vaex special sauce to allow the extreme flexibility for advance pipeline solutions while insuring we
have a tool that works on big data.

* [![Documentation](https://readthedocs.org/projects/vaex/badge/?version=latest)](https://docs.vaex.io)

# Pandas + Sklearn support

Any [Sklearn](https://scikit-learn.org/) + [Pandas](https://pandas.pydata.org/) pipeline/transformer/estimator works as
well.

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
iris = load_iris()
features = iris.feature_names
df = pd.DataFrame(iris.data, columns=features)
df['target'] = iris.target

model = XGBClassifier().fit(df[features], df['target'])
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
pipeline.save( < path >)
pipeline = Pipeline.from_file( < path >)
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

![Alt text](assets/lightgbm-vaex-example.jpg?raw=true "Title")

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
pipeline = Pipeline.from_vaex(df, fit=fit).fit(df)
```

With *Sklearn* the fit would be the standard X and y.

```python
import pandas as pd
import sklearn.pipeline
from sklearn.datasets import load_iris
from xgboost.sklearn import XGBClassifier
from sklearn.pipeline import Pipeline as SklearnPipeline

iris = load_iris()
features = iris.feature_names
df = pd.DataFrame(iris.data, columns=features)
df['target'] = iris.target

# we don't need to provide raw example if we do the training from the Goldilox Pipeline - it would be taken automatically from the first row.
classifier = XGBClassifier(n_estimators=10, verbosity=0, use_label_encoder=False)
pipeline = Pipeline.from_sklearn(classifier).fit(df[features], df['target'])
assert pipeline.validate()

>> > Pipeline
doesn
't handle na for sepal length (cm)
>> > Pipeline
doesn
't handle na for sepal width (cm)
>> > Pipeline
doesn
't handle na for petal length (cm)
>> > Pipeline
doesn
't handle na for petal width (cm)
```

We do not handle missing values? Let's fix that!

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

imputer = ColumnTransformer([('features_mean', SimpleImputer(strategy='mean'),
                              features)], remainder='passthrough')
classifier = XGBClassifier(n_estimators=10, verbosity=0, use_label_encoder=False)

sk_pipeline = sklearn.pipeline.Pipeline([('imputer', imputer), ('classifier', classifier)])
pipeline = Pipeline.from_sklearn(sk_pipeline).fit(df[features], df['target'])
assert pipeline.validate()                              
```

* We can still deploy a pipeline that doesn't deal with missing values if we want, *validate()* returns `True` if
  serialization, and prediction-on-raw validations pass.

# CLI

Some tools

```bash
# Serve model
glx serve <pipeline-path>

# get the variables straight from the file.
glx variables <pipeline-path>

# get the description straight from the file.
glx description <pipeline-path>

# get the raw data example from the file.
glx raw <pipeline-path>

# Get the pipeline requirements
glx freeze <pipeline-path> <path-to-requirements-file-output.txt>

# Get the pipeline requirements
glx install all requirements from the pipeline
```

# Docker

You can build a docker image from a pipeline.

```bash
glx build <pipeline-path> --platform=linux/amd64 --image=python:3.8-slim-bullseye

```

# [Example Notebooks](https://github.com/xdssio/goldilox/tree/master/notebooks)

* **Classification / Regression**
    * [LightGBM](https://github.com/xdssio/goldilox/blob/master/notebooks/lightgbm.ipynb)
    * [XGBoost](https://github.com/xdssio/goldilox/blob/master/notebooks/xgboost.ipynb)
    * [Catbboost](https://github.com/xdssio/goldilox/blob/master/notebooks/catboost.ipynb)
    * [Skleran](https://github.com/xdssio/goldilox/blob/master/notebooks/skleran_simple.ipynb)

* **Clustering**
    * [Kmeans](https://github.com/xdssio/goldilox/blob/master/notebooks/clustering.ipynb)

* **Nearest Neighbours**
    * [KDTree (sklearn)](https://github.com/xdssio/goldilox/blob/master/notebooks/kdtree_nearest_neighbors.ipynb)
    * [hnswlib (recommended)](https://github.com/xdssio/goldilox/blob/master/notebooks/hnswlib_nearest_neighbors.ipynb)
    * [nmslib](https://github.com/xdssio/goldilox/blob/master/notebooks/nmslib_nearest_neighbors.ipynb)
    * [Fiass](https://github.com/xdssio/goldilox/blob/master/notebooks/Fiass_nearest_neighbors.ipynb)

* **Recommendations**
    * [Implicit (Matrix Factorization)](https://github.com/xdssio/goldilox/blob/master/notebooks/implicit.ipynb)

* **Online Learning**
    * [River](https://github.com/xdssio/goldilox/blob/master/notebooks/river_online_learning.ipynb)
    * [Vowpal Wabbit](https://github.com/xdssio/goldilox/blob/master/notebooks/vowpal_wabbit.ipynb)

* **Predictions with Explanations**
    * [SHAP](https://github.com/xdssio/goldilox/blob/master/notebooks/explanations_shap.ipynb)
    * [Interpret](https://github.com/xdssio/goldilox/blob/master/notebooks/interpret.ipynb)

* **NLP**
    * [TFIDF (Sklearn)](https://github.com/xdssio/goldilox/blob/master/notebooks/tfidf.ipynb)
    * [Transformers]() #TODO
    * [Gensim]() #TODO
    * [Spacy]() #TODO
    * [KeyBert]() #TODO

* **Deep Learning**
    * [Keras]() #TODO
    * [Tensorflow]() #TODO
    * [PyTorch]() #TODO
    * [MXNet]() #TODO

* **Training**
    * [AIM](https://github.com/aimhubio/aim) #TODO

* **Advance**
    * [Titanic with feature engineering and LightGBM](https://github.com/xdssio/goldilox/blob/master/notebooks/advance_pipelines.ipynb)
    * [Using a package which is not pickalbe](https://github.com/xdssio/goldilox/blob/master/notebooks/vowpal_wabbit.ipynb)
    * [Imodels](https://github.com/csinva/imodels)
    * [interpret](https://github.com/interpretml/interpret)

# FAQ

1. Why the name "Goldilox"?    
   Because most solutions out there are either tou need to do everything from scratch per solution, or you have to take
   it as it. We consider ourselves in between, you can do most things, with minimal adjustments.
2. Why do you work with Vaex and not just Pandas? Vaex handles Big-Data on normal computers, which is our target
   audience. And we relay heavily on it's lazy evaluation which pandas doesn't have.
3.

# Contributing

See [contributing](https://github.com/xdssio/goldilox/wiki/Contributing) page.

* Notebooks can be a great contribution too!

# Roadmap

See [roadmap](https://github.com/xdssio/goldilox/wiki/Roadmap) page. 
