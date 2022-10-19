from __future__ import annotations

import json
import logging
from copy import deepcopy as _copy
from hashlib import sha256
from tempfile import TemporaryDirectory

import cloudpickle
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted
from typing import List

import goldilox
from goldilox.config import CONSTANTS
from goldilox.utils import is_s3_url, read_bytes, unpickle, validate_path, write_bytes, \
    get_python_version, get_env_type, get_requirements

logger = logging.getLogger()


class Pipeline(TransformerMixin):
    pipeline_type: str
    description: str
    BYTES_SIGNETURE = b"Goldilox"
    BYTE_DELIMITER = b'###'

    @classmethod
    def check_hash(cls, file_path: str) -> int:
        h = sha256()

        with open(file_path, "rb") as file:
            while True:
                # Reading is buffered, so we can read smaller chunks.
                chunk = file.read(h.block_size)
                if not chunk:
                    break
                h.update(chunk)

        return h.hexdigest()

    @staticmethod
    def to_raw(df) -> dict:
        """
        Retrinve
        @param df: a Pandas Dataframe, Pandas Series, Vaex Dataframe, or numpy array.
        @return: a dict example of raw data which you would expect in production inference time.
        """
        if hasattr(df, "to_pandas_df"):  # vaex
            return df.to_records(0)
        elif isinstance(df, np.ndarray):  # numpy
            return list(df[0])
        elif isinstance(df, pd.Series):  # pandas Series
            return {df.name: df[0]}
        return df.iloc[0].to_dict()  # pandas

    def to_pandas(self, df):
        df = self.infer(df)
        if hasattr(df, 'to_pandas_df'):
            df = df.to_pandas_df()
        return df

    @classmethod
    def from_vaex(cls, df,
                  fit=None,
                  predict_column: str = None,
                  variables: dict = None,
                  description: str = "",
                  validate: bool = True) -> Pipeline:
        """
        Get a Pipeline out of a vaex.dataframe.DataFrame, and validate serilization and missing values.
        @param df: vaex.dataframe.DataFrame
        @param fit: method: A method which accepts a vaex dataframe and returns a vaex dataframe which run on pipeline.fit(df).
        @param predict_column: str [optional]: the column to return as values when run predict
        @param variables: dict [optional]: Any variables we want to associate with the current pipeline.
        @param description: str [optional]: Any text we want to associate with the current pipeline.
        @param validate: bool [optional]: If true, run validation.
        @return: VaexPipeline
        """
        from goldilox.vaex.pipeline import VaexPipeline as VaexPipeline
        pipeline = VaexPipeline.from_dataframe(df=df, fit=fit, variables=variables, description=description,
                                               predict_column=predict_column)
        if validate:
            logger.info("validate pipeline")
            logger.info(f"pipeline valid: {pipeline.validate()}")
        return pipeline

    @staticmethod
    def _is_sklearn_fitted(pipeline) -> bool:
        try:
            check_is_fitted(pipeline)
            return True
        except:
            return False

    @classmethod
    def from_sklearn(
            cls,
            pipeline,
            raw: dict = None,
            target: str = None,
            features: List[str] = None,
            output_columns: List[str] = None,
            variables: dict = None,
            fit_params: dict = None,
            description: str = "",
            validate: bool = True
    ) -> Pipeline:
        """
        :param sklearn.preprocessing.pipeline.Pipeline pipeline: The skleran pipeline
        :param raw: dict [optional]: An example of data which will be queried in production (only the features)
                - If X is provided, would be the first row.
        :param features: list [optional]: A list of columns - if X is provided, will take its columns - important if data provided as numpy array.
        :param target: str [optional]: The name of the target column - Used for retraining
        :param output_columns: List[str] [optional]: For sklearn output column in case the output is a numpy array
        :param variables: dict [optional]: Variables to associate with the pipeline - fit_params automatically are added
        :param description: str [optional]: A pipeline description and notes in text
        :param validate: bool [optional]: If True, run validation.
        :return: SkleranPipeline object
        """
        from goldilox.sklearn.pipeline import SklearnPipeline

        ret = SklearnPipeline.from_sklearn(
            pipeline=pipeline,
            features=features,
            target=target,
            raw=raw,
            output_columns=output_columns,
            variables=variables,
            fit_params=fit_params,
            description=description,
        )
        if validate and Pipeline._is_sklearn_fitted(pipeline):
            logger.info("validate pipeline")
            logger.info(f"pipeline valid: {ret.validate()}")
        return ret

    @classmethod
    def _read_pipeline_file(cls, path: str) -> tuple:
        state_bytes = read_bytes(path)
        return Pipeline._split_meta(state_bytes)

    @classmethod
    def from_file(cls, path: str) -> Pipeline:
        """
        Read a pipeline from file.
        @param path: path to pipeline file.
        @return: SkleranPipeline or VaexPipeline.
        """
        meta_bytes, state_bytes = Pipeline._read_pipeline_file(path)
        state = unpickle(state_bytes)
        meta = unpickle(meta_bytes)
        pipeline_type = meta.get(CONSTANTS.PIPELINE_TYPE)
        if pipeline_type == CONSTANTS.SKLEARN:
            return state
        elif pipeline_type == CONSTANTS.VAEX:
            from goldilox.vaex.pipeline import VaexPipeline
            return VaexPipeline.load_state(state)
        raise RuntimeError(f"Cannot load pipeline of type {pipeline_type} from {path}")

    @classmethod
    def load(cls, path: str) -> Pipeline:
        """Alias to from_file()"""
        return cls.from_file(path)

    @classmethod
    def load_meta(cls, path: str) -> dict:
        """Read the meta information from a pipeline file without loading it"""
        meta_bytes, _ = Pipeline._read_pipeline_file(path)
        return unpickle(meta_bytes)

    def _get_meta_dict(self, requirements: List[str] = None, appnope: bool = False) -> dict:
        environment_type = get_env_type()
        return {
            CONSTANTS.PIPELINE_TYPE: self.pipeline_type,
            CONSTANTS.VERSION: goldilox.__version__,
            CONSTANTS.VENV_TYPE: environment_type,
            CONSTANTS.PY_VERSION: get_python_version(),
            CONSTANTS.REQUIREMEMTS: requirements or get_requirements(environment_type, appnope=appnope),
            CONSTANTS.VARIABLES: self.variables.copy(),
            CONSTANTS.DESCRIPTION: self.description,
            CONSTANTS.RAW: self.raw,
        }

    def _get_meta(self, requirements: List[str] = None, appnope: bool = False) -> bytes:
        return cloudpickle.dumps(_copy(self._get_meta_dict(requirements, appnope)))

    @classmethod
    def _split_meta(cls, b: str) -> tuple:
        splited = b.split(Pipeline.BYTE_DELIMITER)
        return splited[0][len(Pipeline.BYTES_SIGNETURE):], splited[1]

    @classmethod
    def _save_state(cls, path: str, state: dict) -> str:
        if not is_s3_url(path):
            validate_path(path)
        return write_bytes(path, state)

    def save(self, path: str, requirements: List[str] = None, **kwargs) -> str:
        """
        @param path: str : output path
        @param requirements: list[str]: a list of requirements. if None - takes from pip automatically
        @param kwargs: Extra parameters to pass
        @return: same path the pipeline was saved to
        """

        state_to_write = Pipeline.BYTES_SIGNETURE + self._get_meta(
            requirements, kwargs.get('appnope')) + Pipeline.BYTE_DELIMITER + self._dumps()
        return self._save_state(path, state_to_write)

    def validate(self, df=None, check_na: bool = True, verbose: bool = True) -> bool:
        """
        Validate the pieline can be saved, reload, and run predictions.
        Can also check if missing value are handled.
        @param df: DataFrame [optional] - used to test prediction.
        @param check_na: bool [optional] - If true, test if missing data is handled.
        @param verbose: If True, log the validation.
        @return: True if pipeline can be served for predictions.
        """
        if verbose:
            logger.info("validate serialization")
        pipeline = Pipeline.from_file(self.save(TemporaryDirectory().name + "models/model.pkl"))
        if df is not None or self.raw is not None:
            X = df if df is not None else self.infer(self.raw)
            if verbose:
                logger.info("validate inference")
            try:
                assert len(pipeline.inference(X)) == len(X)
            except AssertionError as e:
                logger.error("WARNING: Pipeline filter data on inference")
            if check_na:
                pipeline._validate_na()
        elif verbose:
            logger.warning("WARNING: No data provided for inference validation - skip")
        return True

    def _validate_na(self) -> bool:
        ret = True
        copy = self.raw.copy()
        for column in copy:
            tmp = copy.copy()
            tmp[column] = None
            try:
                with np.errstate(all='ignore'):
                    Pipeline.to_raw(self.inference(tmp))
            except Exception as e:
                ret = False
                logger.warning(f"WARNING: Pipeline doesn't handle NA for {column}")
        return ret

    def fit(self, df, **kwargs):
        return self

    def transform(self, df, **kwargs):
        raise NotImplementedError(f"Not implemented for {self.pipeline_type}")

    def predict(self, df, **kwargs):
        raise NotImplementedError(f"Not implemented for {self.pipeline_type}")

    def inference(self, df, **kwargs):
        raise NotImplementedError(f"Not implemented for {self.pipeline_type}")

    def infer(self, df):
        raise NotImplementedError(f"Not implemented for {self.pipeline_type}")

    @classmethod
    def to_records(cls, items) -> List[dict]:
        """Return data as records: [{key: value, ...}, ...]"""
        if isinstance(items, pd.DataFrame):
            return items.to_dict(orient="records")
        elif isinstance(items, (list, dict)):
            return items
        # vaex
        return items.to_records()

    @classmethod
    def jsonify(cls, items) -> List[dict]:
        """Return data as json: '[{key: value, ...}, ...]'"""
        if isinstance(items, pd.DataFrame):
            return items.to_json(orient="records")
        elif isinstance(items, (list, dict)):
            return json.dumps(items)
        # vaex
        return json.dumps(items.to_records())

    def export_mlflow(self, path: str, requirements: List[str] = None, artifacts: dict = None,
                      conda_env: dict = None, input_example: dict = None, signature: dict = None, **kwargs) -> str:
        from goldilox.mlops.mlflow import export_mlflow
        return export_mlflow(self, path=path, requirements=requirements, artifacts=artifacts, conda_env=conda_env,
                             input_example=input_example, signature=signature, **kwargs)

    def export_gunicorn(self, path: str, requirements: List[str] = None,
                        nginx: bool = False, **kwargs) -> str:
        from goldilox.mlops import export_gunicorn
        return export_gunicorn(self, path=path, requirements=requirements, nginx=nginx, **kwargs)

    def export_ray(self, path: str, requirements: List[str] = None, **kwargs) -> str:
        from goldilox.mlops import export_ray
        return export_ray(self, path=path, requirements=requirements, **kwargs)
