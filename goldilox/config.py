import os
AWS_PROFILE = os.environ.get('AWS_PROFILE')

# Constants
STATE = 'state'
CREATED = 'created'
UPDATED = 'updated'
NO_SELECTIONS = {'__filter__': None}
CONSTANTS = [str, int, float, bool]
COLUMNS = 'columns'
VIRTUAL = 'virtual'
OUTPUT_TYPE = 'output_type'
NOT_IMPLEMENTED_ERROR = 'Not implemented'
JSON = 'json'
PANDAS = 'pandas'
VAEX = 'vaex'
SKLEARN = 'sklearn'
NUMPY = 'numpy'
SELECTIONS = 'selections'
SELECTIONS_FILTER = '__filter__'
VARIABLES = 'variables'
PIPELINE = 'pipeline'
PIPELINE_TYPE = 'pipeline_type'
STATE = 'state'
VERSION = 'version'
VALID_VAEX_HEADERS = ['*.csv', '*.hdf5', '*.parquet', '*.arrow']
DEFAULT_SUFFIX = '.parquet'
