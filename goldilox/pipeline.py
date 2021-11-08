import json
from copy import deepcopy
from hashlib import sha256

import cloudpickle
import numpy as np
import pandas as pd

from goldilox.config import AWS_PROFILE, PIPELINE_TYPE
from goldilox.utils import _is_s3_url


class Pipeline:
    pipeline_type: str
    description: str

    @classmethod
    def check_hash(cls, file_path):
        h = sha256()

        with open(file_path, 'rb') as file:
            while True:
                # Reading is buffered, so we can read smaller chunks.
                chunk = file.read(h.block_size)
                if not chunk:
                    break
                h.update(chunk)

        return h.hexdigest()

    @staticmethod
    def _sample(df):
        if hasattr(df, 'to_pandas_df'):  # vaex
            return df.to_records(0)
        elif isinstance(df, np.ndarray):  # numpy
            return list(df[0])
        elif isinstance(df, pd.Series):  # pandas Series
            return {df.name: df[0]}
        return df.iloc[0].to_dict()  # pandas

    @classmethod
    def from_vaex(cls, df, fit=None, **kwargs):
        from goldilox.vaex.pipeline import VaexPipeline as VaexPipeline
        return VaexPipeline.from_dataframe(df=df, fit=fit, **kwargs)

    @classmethod
    def from_sklearn(cls, pipeline, sample=None, target=None, features=None, output_column=None, fit_params=None, **kwargs):
        from goldilox.sklearn.pipeline import SklearnPipeline, DEFAULT_OUTPUT_COLUMN
        output_column = output_column or DEFAULT_OUTPUT_COLUMN
        return SklearnPipeline.from_sklearn(pipeline=pipeline, features=features, target=target, sample=sample,
                                            output_column=output_column, fit_params=fit_params, **kwargs)

    @classmethod
    def load(cls, path):
        return cls.from_file(path)

    @classmethod
    def from_file(self, path):
        if _is_s3_url(path):
            import s3fs
            fs = s3fs.S3FileSystem(profile=AWS_PROFILE)
            with fs.open(path, 'rb') as f:
                state = cloudpickle.loads(f.read())
        else:
            with open(path, 'rb') as f:
                state = cloudpickle.loads(f.read())
        pipeline_type = state.get(PIPELINE_TYPE)
        if pipeline_type == 'sklearn':
            from goldilox.sklearn.pipeline import SklearnPipeline
            return SklearnPipeline.loads(state)
        elif pipeline_type == 'vaex':
            from goldilox.vaex.pipeline import VaexPipeline
            return VaexPipeline.load_state(state)
        raise RuntimeError(f"Cannot load pipeline of type {pipeline_type} from {path}")

    # TODO
    @classmethod
    def _from_koalas(cls, df, **kwargs):
        # from goldilocks.koalas.pipeline import Pipeline as KoalasPipeline
        return deepcopy(df.pipeline)

    # TODO
    @classmethod
    def _from_onnx(self, pipeline, **kwargs):
        raise NotImplementedError(f"Not implemented for {self.pipeline_type}")

    def fit(self, df, **kwargs):
        return self

    def validate(self, df=None, check_na=True):
        raise NotImplementedError(f"Not implemented for {self.pipeline_type}")

    def transform(self, df, **kwargs):
        raise NotImplementedError(f"Not implemented for {self.pipeline_type}")

    def predict(self, df, **kwargs):
        raise NotImplementedError(f"Not implemented for {self.pipeline_type}")

    def inference(self, df, **kwargs):
        raise NotImplementedError(f"Not implemented for {self.pipeline_type}")

    def infer(self, df):
        raise NotImplementedError(f"Not implemented for {self.pipeline_type}")

    @classmethod
    def dictify(cls, items):
        if isinstance(items, pd.DataFrame):
            return items.to_dict(orient='records')
        elif isinstance(items, list) or isinstance(items, dict):
            return items
            # vaex
        return items.to_records()


    @classmethod
    def jsonify(cls, items):
        if isinstance(items, pd.DataFrame):
            return items.to_json(orient='records')
        elif isinstance(items, list) or isinstance(items, dict):
            return json.dumps(items)

        # vaex
        return json.dumps(items.to_records())
