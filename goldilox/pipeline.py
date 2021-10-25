from copy import deepcopy
from hashlib import sha256


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

    @classmethod
    def from_vaex(cls, df, fit=None, warnings=True):
        from goldilox.vaex.pipeline import Pipeline as VaexPipeline
        return VaexPipeline.from_dataframe(df=df, fit=fit, warnings=warnings)

    @classmethod
    def from_pandas(cls, pipeline, X=None, y=None, example=None, features=None, target=None, output_column=None):
        from goldilox.sklearn.pipeline import SklearnPipeline, DEFAULT_OUTPUT_COLUMN
        output_column = output_column or DEFAULT_OUTPUT_COLUMN
        return SklearnPipeline.from_pandas(pipeline=pipeline, X=X, example=example, features=features, target=target,
                                           output_column=output_column)

    @classmethod
    def from_sklearn(cls, pipeline, X=None, target=None, sample=None, features=None, output_column=None):
        from goldilox.sklearn.pipeline import SklearnPipeline, DEFAULT_OUTPUT_COLUMN
        output_column = output_column or DEFAULT_OUTPUT_COLUMN
        return SklearnPipeline.from_sklearn(pipeline=pipeline, features=features, target=target, sample=sample,
                                            output_column=output_column)

    @classmethod
    def _from_koalas(cls, df, **kwargs):
        # from goldilocks.koalas.pipeline import Pipeline as KoalasPipeline
        return deepcopy(df.pipeline)

    def fit(self, df, **kwargs):

        return self

    def transform(self, df, **kwargs):
        raise RuntimeError(f"Not implemented for {self.pipeline_type}")

    def predict(self, df, **kwargs):
        raise RuntimeError(f"Not implemented for {self.pipeline_type}")

    def inference(self, df, **kwargs):
        raise RuntimeError(f"Not implemented for {self.pipeline_type}")

    def infer(self, df):
        raise RuntimeError(f"Not implemented for {self.pipeline_type}")
