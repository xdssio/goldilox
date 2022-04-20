![Logo](https://github.com/xdssio/goldilox/blob/master/assets/logo.png)

# What is Goldilox?

Goldilox is a tool to empower data scientists to build machine learning solutions into production.

* This is in current development, please wait for the first stable version.

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

# Installing

With pip:

```
$ pip install goldilox
```

# Pandas + Sklearn support

Any [Sklearn](https://scikit-learn.org/) + [Pandas](https://pandas.pydata.org/) pipeline/transformer/estimator works can
turn to a pipeline with one line of code, which tou can save and run as a server with the CLI. well.

# Vaex native

[Vaex](https://github.com/vaexio/vaex) is an open-source big data technology with similar APIs
to [Pandas](https://pandas.pydata.org/).   
We use some of Vaex's special sauce to allow the extreme flexibility for advance pipeline solutions while insuring we
have a tool that works on big data.

* [![Documentation](https://readthedocs.org/projects/vaex/badge/?version=latest)](https://docs.vaex.io)

# Examples

**[1. Data science](https://docs.goldilox.io/reference/data-science-examples)**

SKlearn

```python
import pandas as pd
from xgboost.sklearn import XGBClassifier
from goldilox.datasets import load_iris

# Get teh data
df, features, target = load_iris()

# modeling
model = XGBClassifier().fit(df[features], df[target])
```

Vaex

```python
import vaex
from goldilox.datasets import load_iris
from vaex.ml.xgboost import XGBoostModel
import numpy as np

df, features, target = load_iris()
df = vaex.from_pandas(df)

# feature engineering example
df["petal_ratio"] = df["petal_length"] / df["petal_width"]

features.append('petal_ratio')
# modeling
booster = XGBoostModel(
    features=features,
    target=target,
    prediction_name="prediction",
    num_boost_round=500,
)
booster.fit(df)
df = booster.transform(df)

# post modeling processing example 
df["prediction"] = np.around(df["prediction"])
df["label"] = df["prediction"].map({0: "setosa", 1: "versicolor", 2: "virginica"})
```

**2. [Build a production ready pipeline](https://docs.goldilox.io/reference/api-reference/pipeline)**

* In one line (-:

```python
from goldilox import Pipeline

# sklearn - When using sklearn, we want to have an example of the raw production query data
pipeline = Pipeline.from_sklearn(model, raw=Pipeline.to_raw(df[features]))

# vaex
pipeline = Pipeline.from_vaex(df)

# Save and load
pipeline.save( < path >)
pipeline = Pipeline.from_file( < path >)
```

**3. [Deploy](https://docs.goldilox.io/reference/api-reference/cli/serve)**

```
glx serve <path>

[2021-11-16 18:54:44 +0100] [74906] [INFO] Starting gunicorn 20.1.0
[2021-11-16 18:54:44 +0100] [74906] [INFO] Listening at: http://127.0.0.1:5000 (74906)
[2021-11-16 18:54:44 +0100] [74906] [INFO] Using worker: uvicorn.workers.UvicornH11Worker
[2021-11-16 18:54:44 +0100] [74911] [INFO] Booting worker with pid: 74911
[2021-11-16 18:54:44 +0100] [74911] [INFO] Started server process [74911]
[2021-11-16 18:54:44 +0100] [74911] [INFO] Waiting for application startup.
[2021-11-16 18:54:44 +0100] [74911] [INFO] Application startup complete.
```

![Alt text](assets/lightgbm-vaex-example.jpg?raw=true "Title")

**4. [Training](https://docs.goldilox.io/advance/training-re-fitting-todo):**  For experiments, cloud training,
automations, etc,.

With *Vaex*, you put everything you want to do to a function which receives and returns a Vaex DataFrame

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
from sklearn.datasets import load_iris
from xgboost.sklearn import XGBClassifier

iris = load_iris()
features = iris.feature_names
df = pd.DataFrame(iris.data, columns=features)
df['target'] = iris.target

# we don't need to provide raw example if we do the training from the Goldilox Pipeline - it would be taken automatically from the first row.
classifier = XGBClassifier(n_estimators=10, verbosity=0, use_label_encoder=False)
pipeline = Pipeline.from_sklearn(classifier).fit(df[features], df['target'])
```

```
WARNING: Pipeline doesn't handle na for sepal_length
WARNING: Pipeline doesn't handle na for sepal_width
WARNING: Pipeline doesn 't handle na for petal_length
WARNING: Pipeline doesn't handle na for petal_width
```

We do not handle missing values? Let's fix that!

```python
from goldilox.sklearn.transformers import Imputer

classifier = XGBClassifier(n_estimators=10, verbosity=0, use_label_encoder=False)

sk_pipeline = sklearn.pipeline.Pipeline([('imputer', Imputer(features=features)),
                                         ('classifier', classifier)])

pipeline = Pipeline.from_sklearn(sk_pipeline).fit(df[features], df[target])                          
```

* We can still deploy a pipeline that doesn't deal with missing values if we want. Other validations such as
  serialization, and prediction-on-raw must pass.

# [CLI](https://docs.goldilox.io/reference/api-reference/cli)

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

# Update a pipeline file metadata or variables 
glx udpate <pipeline-path> key value --file --variable

```

# Docker

You can build a docker image from a pipeline.

* [Reference](https://docs.goldilox.io/reference/api-reference/cli/build-docker)

```bash
glx build <pipeline-path> --platform=linux/amd64
```

# MLOps

Export to [MLFlow](https://mlflow.org)

```python
pipeline.export_mlflow(path, **kwargs)
```

Export to [Gunicorn](http://gunicorn.org)

```python
pipeline.export_gunicorn(path, **kwargs)
```

# [Data science examples](https://docs.goldilox.io/reference/data-science-examples)

# [Example Notebooks](https://github.com/xdssio/goldilox/tree/master/notebooks)

* **Classification / Regression**
    * [LightGBM](https://github.com/xdssio/goldilox/blob/master/notebooks/lightgbm.ipynb)
    * [XGBoost](https://github.com/xdssio/goldilox/blob/master/notebooks/xgboost.ipynb)
    * [Catbboost](https://github.com/xdssio/goldilox/blob/master/notebooks/catboost.ipynb)
    * [Skleran](https://github.com/xdssio/goldilox/blob/master/notebooks/skleran_simple.ipynb)

* **Clustering**
    * [Kmeans](https://github.com/xdssio/goldilox/blob/master/notebooks/kmeans.ipynb)
    * [hdbscan](https://github.com/xdssio/goldilox/blob/master/notebooks/hdbscan.ipynb)

* **Nearest Neighbours**
    * [KDTree (sklearn)](https://github.com/xdssio/goldilox/blob/master/notebooks/kdtree_nearest_neighbors.ipynb)
    * [hnswlib (recommended)](https://github.com/xdssio/goldilox/blob/master/notebooks/hnswlib_nearest_neighbors.ipynb)
    * [nmslib](https://github.com/xdssio/goldilox/blob/master/notebooks/nmslib_nearest_neighbors.ipynb)
    * [Fiass](https://github.com/xdssio/goldilox/blob/master/notebooks/Fiass_nearest_neighbors.ipynb)

* **Recommendations**
    * [Implicit (Matrix Factorization)](https://github.com/xdssio/goldilox/blob/master/notebooks/implicit.ipynb)
    * [Lightfm (Matrix Factorization with side features)](https://github.com/xdssio/goldilox/blob/master/notebooks/lightfm.ipynb)

* **Online Learning**
    * [River](https://github.com/xdssio/goldilox/blob/master/notebooks/river_online_learning.ipynb)
    * [Vowpal Wabbit](https://github.com/xdssio/goldilox/blob/master/notebooks/vowpal_wabbit.ipynb)

* **Predictions with Explanations**
    * [SHAP](https://github.com/xdssio/goldilox/blob/master/notebooks/explanations_shap.ipynb)
    * [Interpret](https://github.com/xdssio/goldilox/blob/master/notebooks/interpret.ipynb)

* **NLP**
    * [TFIDF (Sklearn)](https://github.com/xdssio/goldilox/blob/master/notebooks/tfidf.ipynb)
    * [Transformers](https://github.com/xdssio/goldilox/blob/master/notebooks/question_answer.ipynb)
    * [Question answering](https://github.com/xdssio/goldilox/blob/master/notebooks/question_answer.ipynb)
    * [Sentiment analysis](https://github.com/xdssio/goldilox/blob/master/notebooks/sentiment_analysis.ipynb)
    * [Spacy](https://github.com/xdssio/goldilox/blob/master/notebooks/entity_extraction_spacy.ipynb)

* **Deep Learning**
    * [PyTorch](https://github.com/xdssio/goldilox/blob/master/notebooks/wide_and_deep.ipynb)
    * [MXNet]() #TODO
    * [Keras]() #TODO
    * [Tensorflow]() #TODO


* **Training**
    * [Retraining](https://github.com/xdssio/goldilox/blob/master/notebooks/training.ipynb)
    * [AIM and Optuna](https://github.com/xdssio/goldilox/blob/master/notebooks/hyperparrameter_optimisation_optuna_aim.ipynb)
    * [MLFlow](https://github.com/xdssio/goldilox/blob/master/notebooks/mlflow.ipynb)
    * [Weights & Biases](https://github.com/xdssio/goldilox/blob/master/notebooks/wandb.ipynb)

* **Advance**
    * [Sklearn vs Vaex vs PySprak](https://github.com/xdssio/goldilox/blob/master/notebooks/sklearn_vs_vaex_vs_pyspark.ipynb)
    * [Using a package which is not pickalbe](https://github.com/xdssio/goldilox/blob/master/notebooks/vowpal_wabbit.ipynb)

# FAQ

1. Why the name "Goldilox"?    
   Because most solutions out there are either tou need to do everything from scratch per solution, or you have to take
   it as it. We consider ourselves in between, you can do most things, with minimal adjustments.
2. Why do you work with Vaex and not just Pandas? Vaex handles Big-Data on normal computers, which is our target
   audience. And we relay heavily on it's lazy evaluation which pandas doesn't have.
3. Why do you use "inference" for predictions and not "predict" or "transform"? Sklearn has a standard, "transform"
   returns a dataframe, "predict" a numpy array, we wanted to have another word for inference. We want the pipeline to
   also follow the sklearn standard with fit, transform, and predict.
4. M1 mac with docker?     
   You probably want to use --platform=linux/amd64
5. How to send arguments to the docker serve?
6. `docker run -p 127.0.0.1:5000:5000 --rm -it --platform=linux/amd64 goldilox glx serve $PIPELINE_PATH <args>`
   example:

* `docker run -p 127.0.0.1:5000:5000 --rm -it --platform=linux/amd64 goldilox glx serve $PIPELINE_PATH --host=0.0.0.0:5000`

# Contributing

See [contributing](https://github.com/xdssio/goldilox/wiki/Contributing) page.

* Notebooks can be a great contribution too!

# Roadmap

See [roadmap](https://github.com/xdssio/goldilox/wiki/Roadmap) page.
