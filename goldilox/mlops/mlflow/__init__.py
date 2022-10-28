from tempfile import TemporaryDirectory
from typing import Union

import goldilox
import goldilox.mlops


def export_mlflow(pipeline: Union[goldilox.Pipeline, str], path: str, artifacts: dict = None,
                  conda_env: dict = None, input_example: dict = None, signature: dict = None, **kwargs) -> str:
    import mlflow.pyfunc
    from mlflow.models import infer_signature

    if isinstance(pipeline, str):
        pipeline = goldilox.Pipeline.from_file(pipeline)
    if conda_env is None:
        conda_env = pipeline.meta.get_conda_environment()

    if artifacts is None:
        artifacts = {}
    pipeline_path = str(TemporaryDirectory().name) + '/pipeline.pkl'
    artifacts['pipeline'] = pipeline.save(pipeline_path)

    class GoldiloxWrapper(mlflow.pyfunc.PythonModel):

        def load_context(self, context):
            self.pipeline = goldilox.Pipeline.from_file(context.artifacts['pipeline'])

        def predict(self, context, model_input):
            return self.pipeline.predict(model_input)

    input_example = input_example or pipeline.raw

    if signature is None:
        data = pipeline.infer(pipeline.raw)
        if hasattr(data, 'to_pandas_df'):
            data = data.to_pandas_df()
        signature = infer_signature(data, pipeline.predict(input_example))

    goldilox.utils.validate_path(path)
    mlflow.pyfunc.save_model(path=path, python_model=GoldiloxWrapper(),
                             artifacts=artifacts,
                             signature=signature,
                             conda_env=conda_env,
                             input_example=input_example,
                             **kwargs
                             )
    return path
