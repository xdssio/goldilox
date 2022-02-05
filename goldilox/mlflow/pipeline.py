import logging

DEFAULT_OUTPUT_COLUMN = "prediction"
TRAITS = "_trait_values"

logger = logging.getLogger()

import mlflow.pyfunc


def export_mlflow(pipeline):
    class MLFlowWrapper(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            from goldilox import Pipeline
            self.pipeline = Pipeline.from_file()
            self.xgb_model.load_model(context.artifacts["pipeline"])

        def predict(self, context, model_input):
            return self.pipeline.inference(model_input)

