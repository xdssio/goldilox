import json
import logging
from hashlib import sha256
from tempfile import TemporaryDirectory

import cloudpickle
import numpy as np
import pandas as pd
import pkg_resources

from goldilox.config import AWS_PROFILE, PIPELINE_TYPE, VAEX, SKLEARN, STATE
from goldilox.utils import _is_s3_url

logger = logging.getLogger()


class Pipeline:
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
        if hasattr(df, "to_pandas_df"):  # vaex
            return df.to_records(0)
        elif isinstance(df, np.ndarray):  # numpy
            return list(df[0])
        elif isinstance(df, pd.Series):  # pandas Series
            return {df.name: df[0]}
        return df.iloc[0].to_dict()  # pandas

    @classmethod
    def from_vaex(cls, df, fit=None, validate=True, **kwargs):
        from goldilox.vaex.pipeline import VaexPipeline as VaexPipeline
        pipeline = VaexPipeline.from_dataframe(df=df, fit=fit, **kwargs)
        if validate:
            logger.info("validate pipeline")
            logger.info(f"pipeline valid: {pipeline.validate()}")
        return pipeline

    @classmethod
    def from_sklearn(
            cls,
            pipeline,
            raw=None,
            target=None,
            features=None,
            output_column=None,
            variables=None,
            fit_params=None,
            description="",
            validate=True
    ):
        from goldilox.sklearn.pipeline import SklearnPipeline, DEFAULT_OUTPUT_COLUMN

        output_column = output_column or DEFAULT_OUTPUT_COLUMN
        ret = SklearnPipeline.from_sklearn(
            pipeline=pipeline,
            features=features,
            target=target,
            raw=raw,
            output_column=output_column,
            variables=variables,
            fit_params=fit_params,
            description=description,
        )
        if validate:
            logger.info("validate pipeline")
            logger.info(f"pipeline valid: {ret.validate()}")
        return ret

    @classmethod
    def from_file(cls, path):
        def open_state():
            return open(path, "rb")

        if _is_s3_url(path):
            import s3fs

            fs = s3fs.S3FileSystem(profile=AWS_PROFILE)

            def open_state():
                return fs.open(path, "rb")

        with open_state() as f:
            state_bytes = f.read()
        state_bytes = Pipeline._remove_signature(state_bytes)
        try:
            state = cloudpickle.loads(state_bytes)
        except:
            import pickle
            logger.warning("issue with cloudpickle loads")
            state = pickle.loads(state_bytes)
        pipeline_type = state.get(PIPELINE_TYPE)
        if pipeline_type == SKLEARN:
            return state[STATE]
        elif pipeline_type == VAEX:
            from goldilox.vaex.pipeline import VaexPipeline
            return VaexPipeline.load_state(state)
        raise RuntimeError(f"Cannot load pipeline of type {pipeline_type} from {path}")

    @classmethod
    def load(cls, path):
        return cls.from_file(path)

    @classmethod
    def _remove_signature(cls, b):
        if b[: len(Pipeline.BYTES_SIGNETURE)] == Pipeline.BYTES_SIGNETURE:
            return b[len(Pipeline.BYTES_SIGNETURE):]
        return b

    def save(self, path):
        state_to_write = Pipeline.BYTES_SIGNETURE + self._dumps()
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
        if verbose:
            logger.info("validate serialization")
        pipeline = Pipeline.from_file(self.save(TemporaryDirectory().name + "models/model.pkl"))
        if df is not None or self.raw is not None:
            if verbose:
                logger.info("validate inference")
            if df is None:
                df = self.infer(self.raw)
            results = pipeline.inference(df)
            assert len(results) == len(df)
            if check_na:
                pipeline._validate_na(df)
        elif verbose:
            logger.info("No data, and no raw example existing for inference validation - skip")
        return True

    # TODO
    @classmethod
    def _from_koalas(cls, df, **kwargs):
        # from goldilocks.koalas.pipeline import Pipeline as KoalasPipeline
        raise NotImplementedError(f"Not implemented for {pipeline.pipeline_type}")

    # TODO
    @classmethod
    def _from_onnx(cls, pipeline, **kwargs):
        raise NotImplementedError(f"Not implemented for {pipeline.pipeline_type}")

    # TODO
    @classmethod
    def _from_mlflow(cls, pipeline, **kwargs):
        raise NotImplementedError(f"Not implemented for {pipeline.pipeline_type}")

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
        if isinstance(items, pd.DataFrame):
            return items.to_dict(orient="records")
        elif isinstance(items, list) or isinstance(items, dict):
            return items
        # vaex
        return items.to_records()

    @classmethod
    def jsonify(cls, items):
        if isinstance(items, pd.DataFrame):
            return items.to_json(orient="records")
        elif isinstance(items, list) or isinstance(items, dict):
            return json.dumps(items)

        # vaex
        return json.dumps(items.to_records())

    @staticmethod
    def _get_packages():
        return sorted(["%s==%s" % (i.key, i.version) for i in pkg_resources.working_set])
