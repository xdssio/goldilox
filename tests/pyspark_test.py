# setup
import os

import numpy as np

os.environ['JAVA_HOME'] = "/Library/Java/JavaVirtualMachines/adoptopenjdk-8.jdk/Contents/Home"
os.environ['JRE_HOME'] = "/Library/java/JavaVirtualMachines/adoptopenjdk-8.jdk/Contents/Home/jre/"
os.environ[
    'PYTHONPATH'] = "/usr/local/Cellar/apache-spark/3.2.0/libexec/python/lib/py4j-0.10.9.2-src.zip:/usr/local/Cellar/apache-spark/3.2.0/libexec/python/:"
os.environ['SPARK_HOME'] = "/usr/local/Cellar/apache-spark/3.2.0/libexec"
os.environ['PYARROW_IGNORE_TIMEZONE'] = "1"
os.environ['PYSPARK_PYTHON'] = "python3"

import findspark

findspark.init()
import pyspark.pandas as ps
from pyspark.sql import SparkSession
import warnings
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

warnings.filterwarnings('ignore')
spark = SparkSession.builder.getOrCreate()

kdf = ps.DataFrame(
    {'a': [1, 2, 3, 4, 5, 6],
     'b': [100, 200, 300, 400, 500, 600],
     'c': ["one", "two", "three", "four", "five", "six"]},
    index=[10, 20, 30, 40, 50, 60])
kdf


def test_pyspark():
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.feature import VectorAssembler

    # Prepare training documents from a list of (id, text, label) tuples.
    train = ps.DataFrame(
        {'feature_a': [1, 2, 3, 4],
         'feature_b': [5, 6, 7, 8],
         'label': [1.0, 0.0, 1.0, 0.0]}).to_spark()

    # Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
    # tokenizer = Tokenizer(inputCol="text", outputCol="words")
    # hashingTF = OneHotEncoder(inputCol='text', outputCol="features")
    assembler = VectorAssembler(inputCols=["feature_a", "feature_b"], outputCol="features")
    lr = LogisticRegression(maxIter=10, regParam=0.001)
    pipeline = Pipeline(stages=[assembler, lr])

    model = pipeline.fit(train)
    model.transform(train).toPandas()
    test = ps.DataFrame(
        {'feature_a': [1, 2, 3, 4],
         "feature_b": [1, 2, 3, 4]}).to_spark()

    prediction = model.transform(test)
    df = prediction.toPandas()
    df[['probability', 'prediction']]

    from onnxmltools import convert_sparkml
    from onnxmltools.convert.sparkml.utils import buildInitialTypesSimple
    initial_types = buildInitialTypesSimple(test)
    onnx_model = convert_sparkml(model, 'Pyspark model', initial_types, spark_session=spark, )

    from tempfile import NamedTemporaryFile
    model_file_path = NamedTemporaryFile().name + '.onnx'
    with open(model_file_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    import onnxruntime as rt
    runtime_test = test.toPandas()
    session = rt.InferenceSession(model_file_path)
    outputs_names = [o.name for o in session.get_outputs()]
    inputs_names = [o.name for o in session.get_inputs()]
    data_in = {o: runtime_test[o].values.astype(np.float32) for o in inputs_names}
    session.get_inputs()[0].type

    pred_onx = session.run(outputs_names, data_in)
