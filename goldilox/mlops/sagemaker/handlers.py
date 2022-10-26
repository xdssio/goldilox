import io
import os

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
        """Loads a model.

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


class PipelineHandlerService(DefaultHandlerService):

    def __init__(self):
        """Initialize a PipelineHandlerService."""
        self._model = None
        self._initialized = False

    def handle(self, data, context):
        """Handle an inference request.

        Args:
            data (obj): the request data.
            context (obj): the request context.

        Returns:
            (bytes, string): data to return to client, response content type.
        """
        if not self._initialized:
            self.initialize(context)

        try:
            request_content_type = context.request_content_type
            return self._model.inference(data)
            # transformer = Transformer(self._model, self._input_fn, self._predict_fn, self._output_fn)
            # return transformer.transform(data, request_content_type, accept)
        except errors.UnsupportedFormatError as e:
            raise e
        except Exception as e:
            raise errors.InternalServerError(str(e))
