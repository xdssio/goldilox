import io
import os
import sys

import pandas as pd

try:  # noqa: FURB107
    from sagemaker_inference import content_types, decoder, default_inference_handler, encoder, errors
    from sagemaker_inference.default_handler_service import DefaultHandlerService
    from sagemaker_inference.transformer import Transformer


except:
    pass

import goldilox
import pathlib

ENABLE_MULTI_MODEL = os.getenv("SAGEMAKER_MULTI_MODEL", "false") == "true"


class InferenceHandler(default_inference_handler.DefaultInferenceHandler):

    def default_model_fn(self, model_dir, context=None):
        """Loads a model. For PyTorch, a default function to load a model cannot be provided.
        Users should provide customized model_fn() in script.

        Args:
            model_dir: a directory where model is saved.
            context (obj): the request context (default: None).

        Returns: A goldilox.Pipeline.
        """
        if pathlib.Path(model_dir).is_file():
            return goldilox.Pipeline.from_file(model_dir)
        return goldilox.Pipeline.from_file(os.path.join(model_dir, 'pipeline.pkl'))

    def default_input_fn(self, input_data, content_type, context=None):
        """A default input_fn that can handle JSON, CSV and NPZ formats.

        Args:
            input_data: the request payload serialized in the content_type format
            content_type: the request content_type
            context (obj): the request context (default: None).

        Returns: input_data deserialized into a dataframe.
        """

        if content_type == 'text/csv':
            return pd.read_csv(io.StringIO(input_data))
        return goldilox.Pipeline.infer(input_data)

    def default_predict_fn(self, data, model, context=None):
        """A default predict_fn for PyTorch. Calls a model on data deserialized in input_fn.
        Runs prediction on GPU if cuda is available.

        Args:
            data: input data (DataFrame) for prediction deserialized by input_fn
            model: A goldilox.Pipeline
            context (obj): the request context (default: None).

        Returns: a prediction
        """
        return model.inference(data)

    def default_output_fn(self, prediction, accept, context=None):
        """A default output_fn for Pipeline. Serializes predictions from predict_fn to JSON, CSV or NPY format.

        Args:
            prediction: a prediction result from predict_fn
            accept: type which the output data needs to be serialized
            context (obj): the request context (default: None).

        Returns: output data serialized
        """
        return goldilox.app.process_response(prediction)


class HandlerService(DefaultHandlerService):
    """Handler service that is executed by the model server.
    Determines specific default inference handlers to use based on model being used.
    This class extends ``DefaultHandlerService``, which define the following:
        - The ``handle`` method is invoked for all incoming inference requests to the model server.
        - The ``initialize`` method is invoked at model server start up.
    Based on: https://github.com/awslabs/multi-model-server/blob/master/docs/custom_service.md
    """

    def __init__(self):
        self._initialized = False

        transformer = Transformer(default_inference_handler=DefaultHandlerService())
        super(HandlerService, self).__init__(transformer=transformer)

    def initialize(self, context):
        # Adding the 'code' directory path to sys.path to allow importing user modules when multi-model mode is enabled.
        if (not self._initialized) and ENABLE_MULTI_MODEL:
            code_dir = os.path.join(context.system_properties.get("model_dir"), 'code')
            sys.path.append(code_dir)
            self._initialized = True

        super().initialize(context)
