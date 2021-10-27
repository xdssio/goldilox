from copy import deepcopy
from hashlib import sha256

from goldilox.config import AWS_PROFILE, PIPELINE_TYPE, STATE
from goldilox.utils import _is_s3_url
import cloudpickle

class Pipeline:
    pipeline_type: str
    example: dict

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
    def _sample_df(df):
        if hasattr(df, 'to_pandas_df'):
            return df.head(1).to_records()[0]
        return df.iloc[0].to_dict()

    @classmethod
    def from_vaex(cls, df, fit=None):
        from goldilox.vaex.pipeline import VaexPipeline as VaexPipeline
        return VaexPipeline.from_dataframe(df=df, fit=fit)

    @classmethod
    def from_sklearn(cls, pipeline, sample=None, target=None, features=None, output_column=None):
        from goldilox.sklearn.pipeline import SklearnPipeline, DEFAULT_OUTPUT_COLUMN
        output_column = output_column or DEFAULT_OUTPUT_COLUMN
        return SklearnPipeline.from_sklearn(pipeline=pipeline, features=features, target=target, sample=sample,
                                            output_column=output_column)



    @classmethod
    def from_file(self, path):
        if _is_s3_url(path):
            import s3fs
            fs = s3fs.S3FileSystem(profile=AWS_PROFILE)
            with fs.open(path, 'r') as f:
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
