import os

from sagemaker_inference import model_server

handler_service = os.getenv('HANDLER_SERVICE', '/opt/program/handler.py')
model_server.start_model_server(handler_service=handler_service)
