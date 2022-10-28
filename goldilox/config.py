import os
from distutils.util import strtobool

AWS_PROFILE = os.environ.get("AWS_PROFILE")


# Constants
class CONSTANTS:
    STATE = "state"
    CREATED = "created"
    UPDATED = "updated"
    NO_SELECTIONS = {"__filter__": None}
    CONSTANTS = [str, int, float, bool]
    COLUMNS = "columns"
    VIRTUAL = "virtual"
    OUTPUT_TYPE = "output_type"
    NOT_IMPLEMENTED_ERROR = "Not implemented"
    JSON = "json"
    PANDAS = "pandas"
    VAEX = "vaex"
    SKLEARN = "sklearn"
    NUMPY = "numpy"
    SELECTIONS = "selections"
    SELECTIONS_FILTER = "__filter__"
    VARIABLES = "variables"
    DESCRIPTION = "description"
    RAW = "raw"
    EXAMPLE = "example"
    ENVIRONMENT_FILENAME = "environment_filename"
    PIPELINE = "pipeline"
    PIPELINE_TYPE = "pipeline_type"
    STATE = "state"
    VERSION = "version"
    GOLDILOX_VERSION = "goldilox_version"
    PY_VERSION = "py_version"
    REQUIREMEMTS = "requirements"
    CONDA_ENV = "conda_env"
    DEFAULT_SUFFIX = ".parquet"
    BYTE_DELIMITER = b'###'
    BYTES_SIGNETURE = b'Goldilox'
    VENV_TYPE = 'venv_type'
    MLOPS = 'mlops'
    CONDA = 'conda'
    VENV = 'venv'
    FASTAPI_PARAMS = 'fastapi_params'


VALID_VAEX_HEADERS = ["*.csv", "*.hdf5", "*.parquet", "*.arrow"]

# App
CORS_ORIGINS = os.getenv('ORIGINS', '*').split(',')
ALLOW_METHODS = os.getenv('ALLOW_METHODS', '*').split(',')
ALLOW_HEADERS = os.getenv('ALLOW_HEADERS', '*').split(',')
ALLOW_CREDENTIALS = bool(strtobool(os.getenv('ALLOW_CREDENTIALS', 'True')))
ALLOW_CORS = bool(strtobool(os.getenv('ALLOW_CORS', 'True')))

# package
GOLDILOX_VERSION = "0.0.23"
