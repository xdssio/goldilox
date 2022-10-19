from tempfile import TemporaryDirectory

from typing import List

from goldilox.utils import validate_path, get_python_version, get_requirements


def export_mlflow(self, path: str, requirements: List[str] = None, artifacts: dict = None,
                  conda_env: dict = None, input_example: dict = None, signature: dict = None, **kwargs) -> str:
    import mlflow.pyfunc
    from mlflow.models import infer_signature
    if conda_env is None:
        if requirements is None:
            _, requirements = get_requirements(venv_type='venv')
            requirements = requirements.split('\n')
        conda_env = {
            'channels': ['defaults'],
            'dependencies': [
                f"python={get_python_version()}",
                {
                    'pip': requirements,
                },
            ],
            'name': 'goldilox_env'
        }

    if artifacts is None:
        pipeline_path = str(TemporaryDirectory().name) + '/pipeline.pkl'
        self.save(pipeline_path)
        artifacts = {"pipeline": pipeline_path}

    class GoldiloxWrapper(mlflow.pyfunc.PythonModel):

        def load_context(self, context):
            from goldilox import Pipeline
            self.pipeline = Pipeline.from_file(context.artifacts['pipeline'])

        def predict(self, context, model_input):
            return self.pipeline.predict(model_input)

    input_example = input_example or self.raw
    if signature is None:
        data = self.infer(input_example)
        if hasattr(data, 'to_pandas_df'):
            data = data.to_pandas_df()
        signature = infer_signature(data, self.predict(input_example))
    validate_path(path)
    mlflow.pyfunc.save_model(path=path, python_model=GoldiloxWrapper(),
                             artifacts=artifacts,
                             signature=signature,
                             conda_env=conda_env,
                             input_example=input_example,
                             **kwargs
                             )
    return path
