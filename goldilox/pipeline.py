import json
import logging
import sys
from copy import deepcopy as _copy
from hashlib import sha256
from tempfile import TemporaryDirectory

import cloudpickle
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted

import goldilox
from goldilox.config import AWS_PROFILE, PIPELINE_TYPE, VAEX, SKLEARN, BYTE_DELIMITER, VERSION, PY_VERSION, \
    PACKAGES, VARIABLES, DESCRIPTION, RAW
from goldilox.utils import _is_s3_url

logger = logging.getLogger()


class Pipeline(TransformerMixin):
    pipeline_type: str
    description: str
    BYTES_SIGNETURE = b"Goldilox"

    @classmethod
    def check_hash(cls, file_path):
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
    def to_raw(df):
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

    @classmethod
    def from_vaex(cls, df, fit=None, variables=None, description="", validate=True):
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
        pipeline = VaexPipeline.from_dataframe(df=df, fit=fit, variables=variables, description=description)
        if validate:
            logger.info("validate pipeline")
            logger.info(f"pipeline valid: {pipeline.validate()}")
        return pipeline

    @staticmethod
    def _is_sklearn_fitted(pipeline):
        try:
            check_is_fitted(pipeline)
            return True
        except:
            return False

    @classmethod
    def from_sklearn(
            cls,
            pipeline,
            raw=None,
            target=None,
            features=None,
            output_columns=None,
            variables=None,
            fit_params=None,
            description="",
            validate=True
    ):
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
    def _read_file(cls, path):
        def open_state():
            return open(path, "rb")

        if _is_s3_url(path):
            import s3fs
            fs = s3fs.S3FileSystem(profile=AWS_PROFILE)

            def open_state():
                return fs.open(path, "rb")

        with open_state() as f:
            state_bytes = f.read()
        return Pipeline._split_meta(state_bytes)

    @classmethod
    def from_file(cls, path):
        """
        Read a pipeline from file.
        @param path: path to pipeline file.
        @return: SkleranPipeline or VaexPipeline.
        """
        meta_bytes, state_bytes = Pipeline._read_file(path)
        try:
            state = cloudpickle.loads(state_bytes)
            meta = cloudpickle.loads(meta_bytes)
        except:
            import pickle
            logger.warning("issue with cloudpickle loads")
            state = pickle.loads(state_bytes)
            meta = pickle.loads(meta_bytes)
        pipeline_type = meta.get(PIPELINE_TYPE)
        if pipeline_type == SKLEARN:
            return state
        elif pipeline_type == VAEX:
            from goldilox.vaex.pipeline import VaexPipeline
            return VaexPipeline.load_state(state)
        raise RuntimeError(f"Cannot load pipeline of type {pipeline_type} from {path}")

    @classmethod
    def load(cls, path):
        """Alias to from_file()"""
        return cls.from_file(path)

    @classmethod
    def load_meta(cls, path):
        """Read the meta information from a pipeline file without loading it"""
        meta_bytes, _ = Pipeline._read_file(path)
        return cloudpickle.loads(meta_bytes)

    def _get_meta(self):
        state = {
            PIPELINE_TYPE: self.pipeline_type,
            VERSION: goldilox.__version__,
            PY_VERSION: sys.version.split(" ")[0],
            PACKAGES: self._get_packages(),
            VARIABLES: self.variables.copy(),
            DESCRIPTION: self.description,
            RAW: self.raw,
        }
        return cloudpickle.dumps(_copy(state))

    @classmethod
    def _split_meta(cls, b):
        splited = b.split(BYTE_DELIMITER)
        return splited[0][len(Pipeline.BYTES_SIGNETURE):], splited[1]

    def save(self, path):
        """Save a pipeline to a file"""
        state_to_write = Pipeline.BYTES_SIGNETURE + self._get_meta() + BYTE_DELIMITER + self._dumps()
        if _is_s3_url(path):
            import s3fs
            fs = s3fs.S3FileSystem(profile=AWS_PROFILE)
            with fs.open(path, "wb") as outfile:
                outfile.write(state_to_write)
        else:
            try:
                import os

                if "/" in path:
                    os.makedirs("/".join(path.split("/")[:-1]), exist_ok=True)
            except AttributeError as e:
                pass
            with open(path, "wb") as outfile:
                outfile.write(state_to_write)

        return path

    def validate(self, df=None, check_na=True, verbose=True):
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

    def _validate_na(self):
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
    def to_records(cls, items):
        """Return data as records: [{key: value, ...}, ...]"""
        if isinstance(items, pd.DataFrame):
            return items.to_dict(orient="records")
        elif isinstance(items, list) or isinstance(items, dict):
            return items
        # vaex
        return items.to_records()

    @classmethod
    def jsonify(cls, items):
        """Return data as json: '[{key: value, ...}, ...]'"""
        if isinstance(items, pd.DataFrame):
            return items.to_json(orient="records")
        elif isinstance(items, list) or isinstance(items, dict):
            return json.dumps(items)
        # vaex
        return json.dumps(items.to_records())

    @staticmethod
    def _get_packages():
        """Run pip freeze and returns the results"""
        import subprocess
        return subprocess.check_output([sys.executable, '-m', 'pip',
                                        'freeze']).decode()
