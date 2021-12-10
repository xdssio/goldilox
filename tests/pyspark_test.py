# setup
import os
os.environ['JAVA_HOME'] = "/Library/Java/JavaVirtualMachines/adoptopenjdk-8.jdk/Contents/Home"
os.environ['JRE_HOME'] = "/Library/java/JavaVirtualMachines/adoptopenjdk-8.jdk/Contents/Home/jre/"
os.environ['PYTHONPATH'] = "/usr/local/Cellar/apache-spark/3.2.0/libexec/python/lib/py4j-0.10.9.2-src.zip:/usr/local/Cellar/apache-spark/3.2.0/libexec/python/:"
os.environ['SPARK_HOME'] = "/usr/local/Cellar/apache-spark/3.2.0/libexec"
# os.environ['ARROW_PRE_0_15_IPC_FORMAT'] = "1"
os.environ['PYARROW_IGNORE_TIMEZONE'] = "1"
os.environ['PYSPARK_PYTHON'] = "python3"

import findspark

findspark.init()
import pyspark.pandas as ps
from pyspark.sql import SparkSession, SQLContext, DataFrame
import warnings
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
    from pyspark.ml.feature import HashingTF, Tokenizer
    from pyspark.sql import SparkSession


    # Prepare training documents from a list of (id, text, label) tuples.
    training = spark.createDataFrame([
        (0, "a b c d e spark", 1.0),
        (1, "b d", 0.0),
        (2, "spark f g h", 1.0),
        (3, "hadoop mapreduce", 0.0)
    ], ["id", "text", "label"])

    # Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
    lr = LogisticRegression(maxIter=10, regParam=0.001)
    pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])

    # Fit the pipeline to training documents.
    model = pipeline.fit(training)

    # Prepare test documents, which are unlabeled (id, text) tuples.
    test = spark.createDataFrame([
        (4, "spark i j k"),
        (5, "l m n"),
        (6, "spark hadoop spark"),
        (7, "apache hadoop")
    ], ["id", "text"])

    # Make predictions on test documents and print columns of interest.
    prediction = model.transform(test)
    selected = prediction.select("id", "text", "probability", "prediction")
    for row in selected.collect():
        rid, text, prob, prediction = row  # type: ignore
        print(
            "(%d, %s) --> prob=%s, prediction=%f" % (
                rid, text, str(prob), prediction  # type: ignore
            )
        )
