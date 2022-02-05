import os.path
import shutil
from tempfile import TemporaryDirectory


def export_mlflow(pipeline, path):
    import mlflow.pyfunc
    from mlflow.models import infer_signature
    env = {
        'channels': ['defaults'],
        'dependencies': [
            f"python={pipeline._get_python_version()}",
            'pip',
            {
                'pip': pipeline._get_packages().split('\n'),
            },
        ],
        'name': 'goldilox_env'
    }

    pipeline_path = str(TemporaryDirectory().name) + '/model.pkl'
    pipeline.save(pipeline_path)
    artifacts = {
        "pipeline": pipeline_path
    }

    class GoldiloxWrapper(mlflow.pyfunc.PythonModel):

        def load_context(self, context):
            from goldilox import Pipeline
            self.pipeline = Pipeline.from_file(context.artifacts['pipeline'])

        def predict(self, context, model_input):
            return self.pipeline.predict(model_input)

    data = pipeline.infer(pipeline.raw)
    if hasattr(data, 'to_pandas_df'):
        data = data.to_pandas_df()
    signature = infer_signature(data, pipeline.predict(pipeline.raw))
    if os.path.isdir(path):
        shutil.rmtree(path)
    mlflow.pyfunc.save_model(path=path, python_model=GoldiloxWrapper(),
                             artifacts=artifacts,
                             signature=signature,
                             conda_env=env)
    return path
