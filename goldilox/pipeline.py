from __future__ import annotations

import json
import logging
import pathlib
from copy import deepcopy as _copy
from hashlib import sha256
from tempfile import TemporaryDirectory
from typing import List

import cloudpickle
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted

import goldilox
from goldilox.config import CONSTANTS
from goldilox.utils import is_s3_url, unpickle, validate_path, read_meta_bytes, remove_signeture, add_signeture

logger = logging.getLogger()


class Pipeline(TransformerMixin):
    pipeline_type: str
    meta: goldilox.Meta

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

    @property
    def variables(self):
        return self.meta.variables

    @property
    def description(self) -> str:
        return self.meta.description

    @property
    def raw(self):
        return self.meta.raw

    def set_raw(self, raw):
        self.meta.raw = raw

    def set_variable(self, key, value):
        self.variables[key] = value
        return value

    def get_variable(self, key, default=None):
        return self.variables.get(key, default)

    @classmethod
    def from_vaex(cls, df,
                  fit=None,
                  variables: dict = None,
                  validate: bool = True,
                  **kwargs) -> Pipeline:
        """
        Get a Pipeline out of a vaex.dataframe.DataFrame, and validate serilization and missing values.
        @param df: vaex.dataframe.DataFrame
        @param fit: method: A method which accepts a vaex dataframe and returns a vaex dataframe which run on pipeline.fit(df).
        @param variables: dict [optional]: Any variables we want to associate with the current pipeline.
        @param description: str [optional]: Any text we want to associate with the current pipeline.
        @param validate: bool [optional]: If true, run validation.
        @return: VaexPipeline
        """
        from goldilox.vaex.pipeline import VaexPipeline as VaexPipeline
        pipeline = VaexPipeline.from_dataframe(df=df, fit=fit, variables=variables, **kwargs)
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
        )
        if validate and Pipeline._is_sklearn_fitted(pipeline):
            logger.info("validate pipeline")
            logger.info(f"pipeline valid: {ret.validate()}")
        return ret

    @classmethod
    def _read_pipeline_file(cls, path: str) -> tuple:
        state_bytes = pathlib.Path(path).read_bytes()
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
        meta_bytes = remove_signeture(read_meta_bytes(path))
        return unpickle(meta_bytes)

    @property
    def _meta_dict(self) -> dict:
        return self.meta.to_dict()

    @property
    def _meta_bytes(self) -> bytes:
        return cloudpickle.dumps(_copy(self._meta_dict))

    @classmethod
    def _split_meta(cls, b: str) -> tuple:
        splited = b.split(CONSTANTS.BYTE_DELIMITER)
        return splited[0][len(CONSTANTS.BYTES_SIGNETURE):], splited[1]

    @classmethod
    def _save_state(cls, path: str, state: dict) -> str:
        if not is_s3_url(path):
            validate_path(path)
        return pathlib.Path(path).write_bytes(state)

    def save(self, path: str) -> str:
        """
        @param path: str : output path
        @param kwargs: Extra parameters to pass
        @return: same path the pipeline was saved to
        """

        state_to_write = add_signeture(self._meta_bytes) + CONSTANTS.BYTE_DELIMITER + self._dumps()
        self._save_state(path, state_to_write)
        return path

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
        if hasattr(items, "to_records"):  # vaex
            return items.to_records()
        if isinstance(items, list):
            return items
        elif isinstance(items, dict):
            return [items]

        return list(items)

    @classmethod
    def jsonify(cls, items) -> List[dict]:
        """Return data as json: '[{key: value, ...}, ...]'"""
        if isinstance(items, pd.DataFrame):
            return items.to_json(orient="records")
        elif hasattr(items, "to_records"):  # vaex
            return json.dumps(items.to_records())

        return json.dumps(items)

    def export_mlflow(self, path: str, artifacts: dict = None,
                      conda_env: dict = None, input_example: dict = None, signature: dict = None, **kwargs) -> str:
        from goldilox.mlops.mlflow import export_mlflow
        return export_mlflow(self, path=path, artifacts=artifacts, conda_env=conda_env,
                             input_example=input_example, signature=signature, **kwargs)

    def export_gunicorn(self, path: str, **kwargs) -> str:
        from goldilox.mlops import export_gunicorn
        return export_gunicorn(self, path=path, **kwargs)

    def export_ray(self, path: str, **kwargs) -> str:
        from goldilox.mlops import export_ray
        return export_ray(self, path=path, **kwargs)
