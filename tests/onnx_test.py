from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tempfile import TemporaryDirectory, NamedTemporaryFile
def test_onnx():
    # Train a model.
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clr = RandomForestClassifier()
    clr.fit(X_train, y_train)

    # Convert into ONNX format
    import pickle
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    initial_type = [('float_input', FloatTensorType([None, 4]))]
    onx = convert_sklearn(clr, initial_types=initial_type)

    pickle.dumps(onx)
    path = NamedTemporaryFile().name+'.onnx'
    with open(path, "wb") as f:
        f.write(onx.SerializeToString())

    # Compute the prediction with ONNX Runtime
    import onnxruntime as rt
    import numpy
    sess = rt.InferenceSession(path)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred_onx = sess.run([label_name], {input_name: X_test.astype(numpy.float32)})[0]

    import onnx
    onnx_model = onnx.load(path)
    onnx.checker.check_model(onnx_model)